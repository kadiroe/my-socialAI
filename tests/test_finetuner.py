import pytest
import torch
from src.models.finetuner import FineTuner
from datasets import Dataset

@pytest.mark.asyncio
async def test_finetuner_init(test_config):
    fine_tuner = FineTuner(test_config)
    assert fine_tuner.model is not None
    assert fine_tuner.tokenizer is not None
    assert fine_tuner.data_collator is not None

def test_prepare_dataset(test_config, test_qa_pairs):
    fine_tuner = FineTuner(test_config)
    dataset = fine_tuner.prepare_dataset(test_qa_pairs)
    
    assert isinstance(dataset, Dataset)
    assert len(dataset) == len(test_qa_pairs)
    assert all(key in dataset.features for key in ['input_ids', 'attention_mask', 'labels'])
    
    # Test for consistent shapes after padding
    first_item = dataset[0]
    assert len(first_item['input_ids']) == 128  # Check padding to max_length
    assert len(first_item['attention_mask']) == 128
    assert len(first_item['labels']) == 128

@pytest.mark.asyncio
async def test_generate_response(test_config):
    fine_tuner = FineTuner(test_config)
    test_question = "What services are available?"
    response = fine_tuner.generate_response(test_question)
    
    assert isinstance(response, str)
    assert len(response) > 0
    
    # Test generation with custom parameters
    response_long = fine_tuner.generate_response(test_question, max_length=512)
    assert isinstance(response_long, str)
    
    # Test error handling
    response_error = fine_tuner.generate_response("")  # Empty question
    assert response_error == "I apologize, but I cannot generate a proper response at this moment."

@pytest.mark.asyncio
async def test_train(test_config, test_qa_pairs, tmp_path):
    from transformers import TrainingArguments, Trainer, TrainerCallback
    import shutil
    import os
    
    # Initialize components
    fine_tuner = FineTuner(test_config)
    dataset = fine_tuner.prepare_dataset(test_qa_pairs)
    output_dir = str(tmp_path / "test_model")
    
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up minimal training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        learning_rate=1e-4,
        logging_steps=1,
        max_steps=2,  # Only run 2 steps
        save_strategy="no",  # Don't save checkpoints in test
        save_total_limit=1,
        dataloader_num_workers=0,  # Single worker
        gradient_accumulation_steps=1,
        logging_strategy="steps",
        disable_tqdm=True,  # Disable progress bars in test
        report_to="none"  # Disable logging
    )
        
    try:
        # Create a minimal training callback that accepts all standard arguments
        class MinimalTrainingCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                # Only process actual training logs
                if logs and 'loss' in logs:
                    current_loss = logs['loss']
                    if current_loss > 100 or (isinstance(current_loss, (int, float)) and torch.isnan(torch.tensor(current_loss))):
                        print(f"Warning: High loss detected ({current_loss}). Training may be unstable.")

        # Create a trainer instance with our callback
        trainer = Trainer(
            model=fine_tuner.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=fine_tuner.data_collator,
            callbacks=[MinimalTrainingCallback()]
        )

        # Perform training
        trainer.train()
        trainer.save_model(output_dir)
        fine_tuner.tokenizer.save_pretrained(output_dir)

        # Verify output directory exists
        assert (tmp_path / "test_model").exists()

        # Check if essential files are saved
        required_files = [
            "adapter_model.safetensors",
            "adapter_config.json"
        ]
        for file in required_files:
            assert (tmp_path / "test_model" / file).exists(), f"Missing required file: {file}"

        # Verify model can generate responses
        test_response = fine_tuner.generate_response("Test question?")
        assert isinstance(test_response, str)
        assert len(test_response) > 0
    
    except Exception as e:
        # Log full error for debugging
        import traceback
        print(f"Training error details:\n{traceback.format_exc()}")
        raise  # Re-raise instead of skip
        
    finally:
        # Cleanup
        if hasattr(fine_tuner, 'trainer'):
            del fine_tuner.trainer
        if os.path.exists(output_dir):
            try:
                shutil.rmtree(output_dir)
            except Exception as e:
                print(f"Error during cleanup: {e}")
            
        # Clean up CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def test_model_cpu_compatibility(test_config):
    fine_tuner = FineTuner(test_config)
    assert next(fine_tuner.model.parameters()).device.type == 'cpu'

def test_optimized_data_collator(test_config):
    fine_tuner = FineTuner(test_config)
    features = [
        {
            "input_ids": [1, 2, 3],
            "attention_mask": [1, 1, 1],
            "labels": [4, 5, 6]
        },
        {
            "input_ids": [7, 8, 9],
            "attention_mask": [1, 1, 1],
            "labels": [10, 11, 12]
        }
    ]
    
    # Test data collator
    batch = fine_tuner.data_collator(features)
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch
    assert batch["input_ids"].dim() == 2  # Should be a 2D tensor
    assert batch["attention_mask"].dim() == 2
    assert batch["labels"].dim() == 2
    assert batch["input_ids"].dtype == torch.int64

def test_training_callback(test_config, capsys):
    from transformers import TrainingArguments, TrainerState, TrainerControl, TrainerCallback
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Create minimal training setup
        fine_tuner = FineTuner(test_config)
        training_args = TrainingArguments(
            output_dir=tmp_dir,
            per_device_train_batch_size=1,
            num_train_epochs=1
        )
        
        # Create a minimal dataset
        dummy_dataset = Dataset.from_dict({
            "input_ids": [[1, 2, 3]],
            "attention_mask": [[1, 1, 1]],
            "labels": [[4, 5, 6]]
        })
        dummy_dataset.set_format("torch")
        
        # Set up trainer with our callback
        class TestCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None):
                if logs and 'loss' in logs:
                    current_loss = logs['loss']
                    if current_loss > 100 or (isinstance(current_loss, (int, float)) and torch.isnan(torch.tensor(current_loss))):
                        print(f"Warning: High loss detected ({current_loss}). Training may be unstable.")
        
        callback = TestCallback()
        state = TrainerState()
        control = TrainerControl()
        
        # Test normal loss
        callback.on_log(training_args, state, control, {"loss": 1.5})
        captured = capsys.readouterr()
        assert "Warning" not in captured.out
        
        # Test high loss
        callback.on_log(training_args, state, control, {"loss": 150.0})
        captured = capsys.readouterr()
        assert "Warning: High loss detected" in captured.out
        
        # Test NaN loss
        callback.on_log(training_args, state, control, {"loss": float('nan')})
        captured = capsys.readouterr()
        assert "Warning: High loss detected" in captured.out
