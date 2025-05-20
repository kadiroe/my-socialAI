from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from typing import List, Dict
import torch
import os

class FineTuner:
    def __init__(self, model_name: str = "facebook/opt-125m"):
        """
        Initialize the finetuner with a base model
        
        Args:
            model_name (str): Name of the base model from Hugging Face
        """
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def prepare_dataset(self, qa_pairs: List[Dict[str, str]]) -> Dataset:
        """
        Convert QA pairs into a dataset for training
        
        Args:
            qa_pairs (List[Dict[str, str]]): List of question-answer pairs
            
        Returns:
            Dataset: Hugging Face dataset ready for training
        """
        # Format data for training
        formatted_data = []
        for pair in qa_pairs:
            text = f"Question: {pair['input']}\nAnswer: {pair['output']}"
            formatted_data.append({"text": text})
            
        return Dataset.from_list(formatted_data)
    
    def train(self, 
              dataset: Dataset,
              output_dir: str = "finetuned_model",
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-5) -> None:
        """
        Finetune the model on the provided dataset
        
        Args:
            dataset (Dataset): Training dataset
            output_dir (str): Directory to save the model
            num_epochs (int): Number of training epochs
            batch_size (int): Training batch size
            learning_rate (float): Learning rate for training
        """
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir="./logs",
        )
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=512
            )
        
        # Prepare dataset
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from datasets import Dataset
from typing import List, Dict
import torch
from ..utils.logging_config import setup_logging

logger = setup_logging(__name__)

class FineTuner:
    def __init__(self, config: dict):
        """Initialize the fine-tuner with configuration."""
        self.config = config['fine_tuner']
        self.model = AutoModelForCausalLM.from_pretrained(self.config['base_model'])
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def prepare_dataset(self, qa_pairs: List[Dict[str, str]]) -> Dataset:
        """Prepare dataset for training."""
        formatted_data = [
            {"text": f"Question: {pair['question']}\nAnswer: {pair['answer']}"}
            for pair in qa_pairs
        ]
        return Dataset.from_list(formatted_data)
    
    def train(self, dataset: Dataset, output_dir: str) -> None:
        """Train the model on the prepared dataset."""
        training_args = TrainingArguments(
            output_dir=output_dir,
            **self.config['training']
        )
        
        tokenized_dataset = dataset.map(
            self._tokenize_function,
            batched=True
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
    def _tokenize_function(self, examples):
        return self.tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=512
        )