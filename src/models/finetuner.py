from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset
from typing import List, Dict
import torch
import os

class FineTuner:
    def __init__(self, config: dict):
        """Initialize the finetuner with configuration"""
        self.config = config['fine_tuner']
        self.model = AutoModelForCausalLM.from_pretrained(self.config['base_model'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create data collator
        self.data_collator = DefaultDataCollator()


    def prepare_dataset(self, qa_pairs: List[Dict[str, str]]) -> Dataset:
        """Convert QA pairs into a dataset for training"""
        formatted_data = []
        for pair in qa_pairs:
            text = f"Question: {pair['question']}\nAnswer: {pair['answer']}"
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_and_format(examples):
            # Tokenize inputs
            tokenized = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Create labels (same as input_ids for causal language modeling)
            labels = tokenized["input_ids"].clone()
            
            # Return dictionary with all required keys
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels  # Add labels for loss computation
            }
        
        # Process dataset
        tokenized_dataset = dataset.map(
            tokenize_and_format,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Set format for PyTorch
        tokenized_dataset.set_format(type="torch")
        
        return tokenized_dataset

    def train(self, dataset: Dataset, output_dir: str) -> None:
        """Train the model on the prepared dataset"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=float(self.config['training']['num_train_epochs']),
            per_device_train_batch_size=int(self.config['training']['per_device_train_batch_size']),
            learning_rate=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay']),
            save_strategy="epoch",
            logging_dir=os.path.join(output_dir, "logs"),
            remove_unused_columns=False,
            # Disable pin memory since no GPU is available
            dataloader_pin_memory=False
        )
    
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.data_collator,  # Use data collator instead of tokenizer
        )
    
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)