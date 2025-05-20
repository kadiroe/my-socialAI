import asyncio
import yaml
import os
from src.data import DatasetProcessor
from src.models import QAGenerator, FineTuner
from src.utils import setup_logging

logger = setup_logging(__name__)

async def main():
    # Load configuration
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize components
    dataset_processor = DatasetProcessor(config)
    qa_generator = QAGenerator(config)
    fine_tuner = FineTuner(config)
    
    # URLs to process
    urls = [
        "https://www.arbeitsagentur.de/",
    ]
    
    try:
        # Create dataset from URLs
        logger.info("Creating dataset from URLs...")
        dataset = await dataset_processor.create_dataset_from_urls(urls)
        if not dataset:
            raise ValueError("Failed to create dataset")
        
        # Generate QA pairs
        logger.info("Generating QA pairs...")
        qa_pairs = []
        for item in dataset:
            pairs = await qa_generator.generate_qa_pairs(item['text'])
            qa_pairs.extend(pairs)
        
        if not qa_pairs:
            raise ValueError("No QA pairs generated")
        
        # Prepare training dataset
        training_dataset = fine_tuner.prepare_dataset(qa_pairs)
        
        # Set up output directory
        output_dir = os.path.join(os.getcwd(), "finetuned_model")
        os.makedirs(output_dir, exist_ok=True)
        
        # Start fine-tuning
        logger.info("Starting model fine-tuning...")
        fine_tuner.train(
            dataset=training_dataset,
            output_dir=output_dir
        )
        logger.info(f"Model fine-tuned and saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        return

if __name__ == "__main__":
    asyncio.run(main())