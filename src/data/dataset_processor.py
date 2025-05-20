from typing import List, Dict, Optional
from datasets import Dataset
import json
import logging
from .web_extractor import WebExtractor
from ..utils.logging_config import setup_logging

logger = setup_logging(__name__)

class DatasetProcessor:
    def __init__(self, config: dict):
        """Initialize dataset processor with configuration."""
        self.web_extractor = WebExtractor(config)
        
    async def create_dataset_from_urls(self, urls: List[str]) -> Optional[Dataset]:
        """
        Create a dataset from a list of URLs.
        
        Args:
            urls (List[str]): List of URLs to process
            
        Returns:
            Optional[Dataset]: Processed dataset or None if failed
        """
        try:
            contents = []
            for url in urls:
                content = await self.web_extractor.get_webpage_content(url)
                if content:
                    contents.append({'text': content, 'source_url': url})
            
            if not contents:
                logger.warning("No content was successfully extracted")
                return None
                
            return Dataset.from_list(contents)
            
        except Exception as e:
            logger.error(f"Error creating dataset: {str(e)}")
            return None
    
    def save_dataset(self, dataset: Dataset, output_path: str) -> bool:
        """
        Save dataset to disk.
        
        Args:
            dataset (Dataset): Dataset to save
            output_path (str): Path to save the dataset
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            dataset.save_to_disk(output_path)
            logger.info(f"Dataset saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving dataset: {str(e)}")
            return False
    
    def load_dataset(self, input_path: str) -> Optional[Dataset]:
        """
        Load dataset from disk.
        
        Args:
            input_path (str): Path to load the dataset from
            
        Returns:
            Optional[Dataset]: Loaded dataset or None if failed
        """
        try:
            dataset = Dataset.load_from_disk(input_path)
            logger.info(f"Dataset loaded from {input_path}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            return None