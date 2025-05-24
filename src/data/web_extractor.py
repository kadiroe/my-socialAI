import requests
from bs4 import BeautifulSoup
from typing import Optional, Set, Dict, List
import logging
from urllib.parse import urlparse, urljoin
import asyncio
import aiohttp
from collections import defaultdict
from ..utils.logging_config import setup_logging

logger = setup_logging(__name__)

class WebExtractor:
    def __init__(self, config: dict):
        """Initialize the web extractor with configuration."""
        self.config = config['web_extractor']
        self.headers = {
            'User-Agent': self.config['user_agent']
        }
        self.min_line_length = self.config['min_line_length']
        self.visited_urls = set()
        self.content_by_url = defaultdict(str)

    async def get_webpage_content(self, url: str) -> Optional[str]:
        """
        Fetch and extract content from a webpage and its subdomains/linked pages.
        
        Args:
            url (str): Starting URL to crawl
            
        Returns:
            Optional[str]: Combined extracted content or None if failed
        """
        if not self._is_valid_url(url):
            logger.error(f"Invalid URL provided: {url}")
            return None

        try:
            base_domain = self._get_base_domain(url)
            self.visited_urls.clear()
            self.content_by_url.clear()

            # Create a session for connection pooling
            async with aiohttp.ClientSession(headers=self.headers) as session:
                await self._crawl_url(session, url, base_domain, depth=0)

            # Combine all content
            combined_content = "\n\n".join(self.content_by_url.values())
            return combined_content if combined_content.strip() else None

        except Exception as e:
            logger.error(f"Error in web extraction process: {str(e)}")
            return None

    async def _crawl_url(self, session: aiohttp.ClientSession, url: str, base_domain: str, depth: int) -> None:
        """Recursively crawl URLs up to specified depth."""
        if (
            depth > self.config['crawling']['max_depth'] or
            url in self.visited_urls or
            len(self.visited_urls) >= self.config['crawling']['max_pages_per_domain']
        ):
            return

        self.visited_urls.add(url)
        
        try:
            # Fetch and parse content
            async with session.get(url, timeout=self.config['crawling']['timeout']) as response:
                if response.status != 200:
                    return
                html = await response.text()
                
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract main content
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
            
            main_content = (
                soup.find('main') or 
                soup.find('article') or 
                soup.find('div', class_=['content', 'main'])
            )
            
            content = main_content.get_text(separator=' ', strip=True) if main_content else soup.body.get_text(separator=' ', strip=True)
            cleaned_content = self._clean_content(content)
            
            if cleaned_content:
                self.content_by_url[url] = cleaned_content

            # Find and process links
            links = self._extract_links(soup, url, base_domain)
            
            # Process links concurrently with rate limiting
            tasks = []
            sem = asyncio.Semaphore(self.config['crawling']['concurrent_requests'])
            
            for link in links:
                if link not in self.visited_urls:
                    tasks.append(self._bounded_crawl(sem, session, link, base_domain, depth + 1))
            
            if tasks:
                await asyncio.gather(*tasks)
                
        except Exception as e:
            logger.error(f"Error crawling {url}: {str(e)}")

    async def _bounded_crawl(self, sem, session, url, base_domain, depth):
        """Crawl with a semaphore for rate limiting."""
        async with sem:
            await self._crawl_url(session, url, base_domain, depth)

    def _extract_links(self, soup: BeautifulSoup, current_url: str, base_domain: str) -> Set[str]:
        """Extract relevant links from the page."""
        links = set()
        current_domain = self._get_base_domain(current_url)
        
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            absolute_url = urljoin(current_url, href)
            
            if not self._is_valid_url(absolute_url):
                continue
                
            link_domain = self._get_base_domain(absolute_url)
            
            # Check if link is in same domain or subdomain
            if (link_domain == base_domain or 
                (self.config['crawling']['include_subdomains'] and 
                 link_domain.endswith(f".{base_domain}"))):
                links.add(absolute_url)
        
        return links

    def _get_base_domain(self, url: str) -> str:
        """Extract base domain from URL."""
        parsed = urlparse(url)
        parts = parsed.netloc.split('.')
        
        # Handle special cases like co.uk
        if len(parts) > 2 and parts[-2] in ['co', 'com', 'org', 'gov']:
            return '.'.join(parts[-3:])
        return '.'.join(parts[-2:])

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