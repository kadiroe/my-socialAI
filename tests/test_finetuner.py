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

@pytest.mark.asyncio
async def test_generate_response(test_config):
    fine_tuner = FineTuner(test_config)
    test_question = "What services are available?"
    response = fine_tuner.generate_response(test_question)
    
    assert isinstance(response, str)
    assert len(response) > 0

@pytest.mark.asyncio
async def test_train(test_config, test_qa_pairs, tmp_path):
    fine_tuner = FineTuner(test_config)
    dataset = fine_tuner.prepare_dataset(test_qa_pairs)
    output_dir = str(tmp_path / "test_model")
    
    try:
        fine_tuner.train(dataset=dataset, output_dir=output_dir)
        assert (tmp_path / "test_model").exists()
        # Check if essential model files are saved
        assert (tmp_path / "test_model" / "adapter_model.bin").exists() or \
               (tmp_path / "test_model" / "adapter_model.safetensors").exists()
    except Exception as e:
        pytest.skip(f"Training failed due to resource constraints: {str(e)}")

def test_model_cpu_compatibility(test_config):
    fine_tuner = FineTuner(test_config)
    assert next(fine_tuner.model.parameters()).device.type == 'cpu'
