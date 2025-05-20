import os
from dotenv import load_dotenv
import google.genai as genai

def setup_api():
    """
    Initialize the Gemini API with credentials from environment variables.
    """
    load_dotenv()  # Load environment variables from .env file
    
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)