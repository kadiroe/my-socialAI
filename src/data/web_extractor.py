import requests
from bs4 import BeautifulSoup
from typing import Optional
import logging
from urllib.parse import urlparse
from ..utils.logging_config import setup_logging

logger = setup_logging(__name__)

class WebExtractor:
    def __init__(self, config: dict):
        self.headers = {
            'User-Agent': config['web_extractor']['user_agent']
        }
        self.min_line_length = config['web_extractor']['min_line_length']

    async def get_webpage_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract main content from a webpage.
        
        Args:
            url (str): URL of the webpage to scrape
            
        Returns:
            Optional[str]: Extracted content or None if failed
        """
        try:
            if not self._is_valid_url(url):
                raise ValueError("Invalid URL provided")

            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_=['content', 'main'])
            )
            
            content = main_content.get_text(separator=' ', strip=True) if main_content else soup.body.get_text(separator=' ', strip=True)
            return self._clean_content(content)
            
        except Exception as e:
            logger.error(f"Error fetching webpage: {str(e)}")
            return None
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _clean_content(self, content: str) -> str:
        """Clean extracted content."""
        content = ' '.join(content.split())
        lines = [line for line in content.split('\n') if len(line.strip()) > self.min_line_length]
        return '\n'.join(lines)