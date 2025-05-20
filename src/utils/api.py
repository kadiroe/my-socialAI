import os
from dotenv import load_dotenv
from google import genai

def setup_api() -> genai.Client:
    """
    Initialize and return the Gemini API client.
    
    Returns:
        genai.Client: Configured Gemini client
    """
    load_dotenv()
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    return genai.Client(api_key=api_key)