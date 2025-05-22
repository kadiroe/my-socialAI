from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DefaultDataCollator
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

class FineTuner:
    def __init__(self, config: dict):
        """Initialize the finetuner with configuration"""
        self.config = config['fine_tuner']
        
        # Load model with CPU configuration
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Move model to CPU explicitly
        self.model = self.model.to('cpu')
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['base_model'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA config to model
        self.model = get_peft_model(self.model, lora_config)
        
        # Create data collator
        self.data_collator = DefaultDataCollator()

    def prepare_dataset(self, qa_pairs: List[Dict[str, str]]) -> Dataset:
        """Convert QA pairs into a dataset for training"""
        formatted_data = []
        for pair in qa_pairs:
            # Format as instruction following TinyLlama-Chat template
            text = f"<|system|>You are a helpful assistant that provides accurate and relevant answers.</s><|user|>{pair['question']}</s><|assistant|>{pair['answer']}</s>"
            formatted_data.append({"text": text})
        
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_and_format(examples):
            # Tokenize inputs with smaller max_length for CPU memory constraints
            tokenized = self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=256,  # Reduced from 512 for CPU memory
                return_tensors="pt"
            )
            
            labels = tokenized["input_ids"].clone()
            return {
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "labels": labels
            }
        
        tokenized_dataset = dataset.map(
            tokenize_and_format,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        tokenized_dataset.set_format(type="torch")
        return tokenized_dataset

    def train(self, dataset: Dataset, output_dir: str) -> None:
        """Train the model on the prepared dataset"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=float(self.config['training']['num_train_epochs']),
            per_device_train_batch_size=2,  # Reduced batch size for CPU
            gradient_accumulation_steps=8,  # Increased for effective batch size
            learning_rate=float(self.config['training']['learning_rate']),
            weight_decay=float(self.config['training']['weight_decay']),
            max_grad_norm=float(self.config['training']['max_grad_norm']),
            warmup_ratio=float(self.config['training']['warmup_ratio']),
            lr_scheduler_type=self.config['training']['lr_scheduler_type'],
            save_strategy="steps",
            save_steps=25,
            logging_steps=5,
            optim="adamw_torch",
            remove_unused_columns=False,
            dataloader_pin_memory=False,
            # Force CPU training
            no_cuda=True,
            use_cpu=True
        )
    
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=self.data_collator,
        )
    
        trainer.train()
        
        # Save trained model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
    def generate_response(self, question: str, max_length: int = 256) -> str:
        """Generate a response using the fine-tuned model"""
        prompt = f"<|system|>You are a helpful assistant that provides accurate and relevant answers.</s><|user|>{question}</s><|assistant|>"
        
        inputs = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        # Ensure inputs are on CPU
        inputs = {k: v.to('cpu') for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_beams=2,  # Reduced for CPU
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("</s>")
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("<|assistant|>")[-1].strip()
        return response