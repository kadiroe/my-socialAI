from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForSeq2Seq,
    TrainerCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import Dataset
from typing import List, Dict, Optional
import torch
import os
import gc
import numpy as np

class FineTuner:
    def __init__(self, config: dict):
        """Initialize the finetuner with configuration"""
        self.config = config['fine_tuner']
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Load model with training configuration
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config['base_model'],
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
            use_cache=False  # Disable cache since we're training
        )
        
        # Move model to CPU
        self.model = self.model.to('cpu')
        
        # Enable gradient computation explicitly
        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            use_fast=True,
            model_max_length=128  # Reduced for memory
        )
            
        # Configure LoRA
        lora_config = LoraConfig(
            r=self.config['lora']['r'],
            lora_alpha=self.config['lora']['lora_alpha'],
            target_modules=self.config['lora']['target_modules'],
            lora_dropout=self.config['lora']['lora_dropout'],
            bias=self.config['lora']['bias'],
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False
        )
        
        # Apply LoRA and prepare for training
        self.model = get_peft_model(self.model, lora_config)
        
        # Create custom data collator for better performance
        class OptimizedDataCollator(DataCollatorForSeq2Seq):
            def __call__(self, features, return_tensors=None):
                # Pre-convert lists to numpy arrays
                if features and "labels" in features[0]:
                    labels = np.array([f["labels"] for f in features], dtype=np.int64)
                    input_ids = np.array([f["input_ids"] for f in features], dtype=np.int64)
                    attention_mask = np.array([f["attention_mask"] for f in features], dtype=np.int64)
                    
                    # Convert numpy arrays to tensors
                    batch = {
                        "input_ids": torch.from_numpy(input_ids),
                        "attention_mask": torch.from_numpy(attention_mask),
                        "labels": torch.from_numpy(labels)
                    }
                    return batch
                return super().__call__(features, return_tensors)
        
        # Initialize optimized data collator
        self.data_collator = OptimizedDataCollator(
            tokenizer=self.tokenizer,
            model=self.model,
            padding="longest",
            max_length=128,
            pad_to_multiple_of=8
        )
        
        # Verify trainable parameters
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}")
        
        # Initialize trainer as None
        self.trainer = None
        
        # Clear memory
        gc.collect()

    def prepare_dataset(self, qa_pairs: List[Dict[str, str]]) -> Dataset:
        """Convert QA pairs into a dataset for training"""
        formatted_data = []
        
        # Process all pairs at once since we have reduced memory usage elsewhere
        for pair in qa_pairs:
            formatted_data.append({
                "input_text": f"question: {pair['question']}",
                "target_text": pair['answer']
            })
        
        dataset = Dataset.from_list(formatted_data)
        
        def tokenize_function(examples):
            # Tokenize inputs with padding
            model_inputs = self.tokenizer(
                examples["input_text"],
                padding="max_length",  # Changed to max_length padding
                truncation=True,
                max_length=128,
                return_tensors=None,  # Return lists instead of tensors
            )
            
            # Tokenize targets with padding
            with torch.no_grad():
                labels = self.tokenizer(
                    examples["target_text"],
                    padding="max_length",  # Changed to max_length padding
                    truncation=True,
                    max_length=128,
                    return_tensors=None,  # Return lists instead of tensors
                )
            
            # Convert to numpy array after ensuring consistent lengths
            model_inputs["labels"] = np.array(labels["input_ids"], dtype=np.int64)
            model_inputs["input_ids"] = np.array(model_inputs["input_ids"], dtype=np.int64)
            model_inputs["attention_mask"] = np.array(model_inputs["attention_mask"], dtype=np.int64)
            return model_inputs
        
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            batch_size=len(dataset),
            remove_columns=dataset.column_names,
            desc="Tokenizing dataset"
        )
        
        # Force PyTorch format
        tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        return tokenized_dataset

    def train(self, dataset: Dataset, output_dir: str) -> None:
        """Train the model on the prepared dataset with improved error handling and optimizations"""
        try:
            # Clear memory before training
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Ensure model is in training mode
            self.model.train()
            
            # Calculate optimal training parameters
            dataset_size = len(dataset)
            effective_batch_size = self.config['training'].get('per_device_train_batch_size', 4)
            grad_accum_steps = max(1, 16 // effective_batch_size)  # Target effective batch size of 16
            
            # Set up training arguments with compatible parameters
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=float(self.config['training']['num_train_epochs']),
                per_device_train_batch_size=effective_batch_size,
                gradient_accumulation_steps=grad_accum_steps,
                learning_rate=float(self.config['training']['learning_rate']),
                weight_decay=float(self.config['training']['weight_decay']),
                max_grad_norm=float(self.config['training']['max_grad_norm']),
                warmup_ratio=float(self.config['training']['warmup_ratio']),
                lr_scheduler_type=self.config['training']['lr_scheduler_type'],
                save_strategy="epoch",
                logging_steps=max(10, dataset_size // (effective_batch_size * 5)),  # Log ~5 times per epoch
                save_total_limit=2,
                remove_unused_columns=False,
                push_to_hub=False,
                disable_tqdm=False,
                gradient_checkpointing=True,
                # Basic optimizations that should work across versions
                fp16=False,  # Disable mixed precision on CPU
                group_by_length=True,  # Reduce padding by grouping similar lengths
                dataloader_pin_memory=False  # Disable pin_memory since we're on CPU
            )

            # Initialize trainer with improved error handling
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=dataset,
                data_collator=self.data_collator,
            )
            
            # Add training hooks for better monitoring using a proper callback class
            class TrainingCallback(TrainerCallback):
                def on_log(self, args, state, control, logs=None):
                    if logs and 'loss' in logs:
                        current_loss = logs['loss']
                        if current_loss > 100 or (isinstance(current_loss, (int, float)) and torch.isnan(torch.tensor(current_loss))):
                            print(f"Warning: High loss detected ({current_loss}). Training may be unstable.")
            
            self.trainer.add_callback(TrainingCallback())
            
            # Train the model with error handling
            try:
                self.trainer.train()
            except Exception as e:
                print(f"Training error occurred: {str(e)}")
                if "CUDA out of memory" in str(e):
                    print("Memory error detected. Try reducing batch size or model size.")
                elif "loss is not finite" in str(e):
                    print("Training diverged. Try reducing learning rate or increasing warmup steps.")
                raise
            
            # Save the final model
            self.trainer.save_model(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            
            # Clear memory after training
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Fatal error during training: {str(e)}")
            raise
        
        finally:
            # Ensure cleanup happens
            gc.collect()

    def generate_response(self, question: str, max_length: int = 256) -> str:
        """Generate a response using the fine-tuned model"""
        try:
            # Prepare model for inference
            self.model.eval()
            
            # Check for empty input
            if not question.strip():
                return "I apologize, but I cannot generate a proper response at this moment."
                
            # Format input with better prompt template
            input_text = f"Answer the following question clearly and thoroughly: {question}"
            inputs = self.tokenizer(
                input_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512  # Allow longer input
            )
            
            # Generate response with improved parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=50,  # Encourage more detailed responses
                    num_beams=4,  # Increased for better quality
                    temperature=0.8,  # Slightly increased creativity
                    top_p=0.92,  # Slightly increased diversity
                    top_k=50,  # Added top-k filtering
                    repetition_penalty=1.3,  # Increased repetition penalty
                    length_penalty=1.5,  # Encourage longer responses
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Basic post-processing
            response = response.strip()
            if not response:  # Fallback for empty responses
                return "I apologize, but I cannot generate a proper response at this moment."
                
            return response
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "An error occurred while generating the response."
            
        finally:
            # Ensure model is back in training mode
            self.model.train()