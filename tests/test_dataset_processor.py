import pytest
from src.data.dataset_processor import DatasetProcessor

@pytest.mark.asyncio
async def test_dataset_processor_init(test_config):
    processor = DatasetProcessor(test_config)
    assert processor.web_extractor is not None

@pytest.mark.asyncio
async def test_create_dataset_from_urls(test_config):
    processor = DatasetProcessor(test_config)
    urls = ["https://example.com"]
    dataset = await processor.create_dataset_from_urls(urls)
    # If website is unreachable, should return None
    assert dataset is None or hasattr(dataset, 'map')

@pytest.mark.asyncio
async def test_save_and_load_dataset(test_config, tmp_path):
    processor = DatasetProcessor(test_config)
    test_data = [{"text": "Test content", "source_url": "https://example.com"}]
    
    from datasets import Dataset
    dataset = Dataset.from_list(test_data)
    
    # Test save
    output_path = tmp_path / "test_dataset"
    save_result = processor.save_dataset(dataset, str(output_path))
    assert save_result is True
    
    # Test load
    loaded_dataset = processor.load_dataset(str(output_path))
    assert loaded_dataset is not None
    assert len(loaded_dataset) == len(test_data)
    assert loaded_dataset[0]["text"] == test_data[0]["text"]
