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
        logger.info(f"Successfully created dataset with {len(dataset)} documents")
        
        # Generate QA pairs
        logger.info("Generating QA pairs...")
        qa_pairs = []
        for idx, item in enumerate(dataset, 1):
            if 'text' not in item:
                logger.warning(f"Document {idx} missing 'text' field")
                continue
                
            pairs = await qa_generator.generate_qa_pairs(item['text'])
            if pairs:
                qa_pairs.extend(pairs)
                logger.info(f"Generated {len(pairs)} QA pairs from document {idx}")
        
        if not qa_pairs:
            raise ValueError("No QA pairs generated")
        logger.info(f"Total QA pairs generated: {len(qa_pairs)}")
        
        # Prepare training dataset
        logger.info("Preparing training dataset...")
        training_dataset = fine_tuner.prepare_dataset(qa_pairs)
        logger.info(f"Training dataset prepared with {len(training_dataset)} examples")
        
        # Set up output directory
        output_dir = os.path.join(os.getcwd(), "finetuned_model")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory at {output_dir}")
        
        # Start fine-tuning
        logger.info("Starting model fine-tuning...")
        fine_tuner.train(
            dataset=training_dataset,
            output_dir=output_dir
        )
        logger.info(f"Model fine-tuned and saved to {output_dir}")
        
        # Test the model with some example questions
        logger.info("Testing the fine-tuned model...")
        test_questions = [
            "What services does the Arbeitsagentur offer?",
            "How can I register as unemployed?",
            "What documents do I need for job seeking?"
        ]
        
        for question in test_questions:
            logger.info(f"\nQuestion: {question}")
            response = fine_tuner.generate_response(question)
            logger.info(f"Response: {response}\n")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise  # Re-raise the exception for debugging
        
if __name__ == "__main__":
    asyncio.run(main())