import google.genai as genai
from typing import List, Dict
from ..utils.logging_config import setup_logging
from ..utils.api import setup_api

logger = setup_logging(__name__)

class QAGenerator:
    def __init__(self, config: dict):
        """Initialize QA Generator with configuration."""
        setup_api()  # Initialize API configuration
        self.model = genai.GenerativeModel(config['qa_generator']['model_name'])
        self.num_default_pairs = config['qa_generator']['num_default_pairs']
    
    async def generate_qa_pairs(self, context: str, num_pairs: int = None) -> List[Dict[str, str]]:
        """
        Generate question-answer pairs based on given context.
        
        Args:
            context (str): The context text
            num_pairs (int, optional): Number of QA pairs to generate
            
        Returns:
            List[Dict[str, str]]: List of QA pairs
        """
        num_pairs = num_pairs or self.num_default_pairs
        prompt = self._create_prompt(context, num_pairs)
        
        try:
            response = await self.model.generate_content(prompt)
            return self._parse_response(response.text)
        except Exception as e:
            logger.error(f"Error generating QA pairs: {str(e)}")
            return []
    
    def _create_prompt(self, context: str, num_pairs: int) -> str:
        return f"""
        Generate {num_pairs} unique question-answer pairs from this context:
        
        {context}
        
        Format:
        Question: <question>
        Answer: <answer>
        """
    
    def _parse_response(self, response_text: str) -> List[Dict[str, str]]:
        qa_pairs = []
        pairs = response_text.split('\n\n')
        
        for pair in pairs:
            if 'Question:' in pair and 'Answer:' in pair:
                question = pair.split('Answer:')[0].replace('Question:', '').strip()
                answer = pair.split('Answer:')[1].strip()
                qa_pairs.append({'question': question, 'answer': answer})
        
        return qa_pairs