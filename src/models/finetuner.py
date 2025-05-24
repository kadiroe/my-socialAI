from transformers import (
    AutoModelForSeq2SeqLM,  # Changed for T5
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq  # Changed for T5
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)
from datasets import Dataset
from typing import List, Dict, Optional
import torch
import os
import gc

class FineTuner:
    def __init__(self, config: dict):
        """Initialize the finetuner with configuration"""
        self.config = config['fine_tuner']
        
        # Clear CUDA cache and garbage collect
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Load model with minimal memory footprint
        self.model = AutoModelForSeq2SeqLM.from_pretrained(  # Changed for T5
            self.config['base_model'],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_cache=False,
            trust_remote_code=True
        )
        
        # Move model to CPU explicitly and clear memory
        self.model = self.model.to('cpu')
        gc.collect()
        
        # Load tokenizer with minimal settings
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            use_fast=True,
            model_max_length=256
        )
            
        # Configure LoRA with minimal parameters
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.SEQ_2_SEQ_LM,  # Changed for T5
            inference_mode=False,
            modules_to_save=None
        )
        
        # Apply LoRA config to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Create data collator specific for T5
        self.data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Free memory
        torch.cuda.empty_cache()
        gc.collect()

    def prepare_dataset(self, qa_pairs: List[Dict[str, str]]) -> Dataset:
        """Convert QA pairs into a dataset for training"""
        formatted_data = []
        
        # Process in smaller chunks to save memory
        chunk_size = 10
        for i in range(0, len(qa_pairs), chunk_size):
            chunk = qa_pairs[i:i + chunk_size]
            for pair in chunk:
                # Format for T5: "question: {question} answer: {answer}"
                input_text = f"question: {pair['question']}"
                target_text = pair['answer']
                formatted_data.append({
                    "input_text": input_text,
                    "target_text": target_text
                })
            
            # Clear memory after each chunk
            gc.collect()
        
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_and_format(examples):
            # Tokenize inputs and targets separately
            model_inputs = self.tokenizer(
                examples["input_text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Tokenize targets
            labels = self.tokenizer(
                examples["target_text"],
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs
        
        # Process dataset in smaller batches
        tokenized_dataset = dataset.map(
            tokenize_and_format,
            batched=True,
            batch_size=4,
            remove_columns=dataset.column_names
        )
        
        tokenized_dataset.set_format(type="torch")
        return tokenized_dataset

    def train(self, dataset: Dataset, output_dir: str) -> None:
        """Train the model on the prepared dataset"""
        # Clear memory before training
        torch.cuda.empty_cache()
        gc.collect()
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=float(self.config['training']['num_train_epochs']),
            per_device_train_batch_size=2,  # Slightly increased for T5
            gradient_accumulation_steps=8,  # Reduced for T5
            learning_rate=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay']),
            max_grad_norm=float(self.config['training']['max_grad_norm']),
            warmup_ratio=float(self.config['training']['warmup_ratio']),
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            save_strategy="steps",
            save_steps=50,
            logging_steps=10,
            optim="adamw_torch",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            no_cuda=True,
            use_cpu=True,
            gradient_checkpointing=True,
            eval_steps=None,
            save_total_limit=2
        )
    
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.data_collator,
        )
    
        # Train with memory optimization
        trainer.train()
        
        # Save trained model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Clear memory after training
        torch.cuda.empty_cache()
        gc.collect()
        
    def generate_response(self, question: str, max_length: int = 128) -> str:
        """Generate a response using the fine-tuned model"""
        # Clear memory before generation
        torch.cuda.empty_cache()
        gc.collect()
        
        # Format input for T5
        input_text = f"question: {question}"
        
        inputs = self.tokenizer(input_text, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=2,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                use_cache=False
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clear memory after generation
        torch.cuda.empty_cache()
        gc.collect()
        
        return response