import pytest
import os
from src.utils.logging_config import setup_logging
from src.utils.api import setup_api

def test_setup_logging():
    logger = setup_logging("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == 20  # INFO level
    
    # Test with different level
    logger = setup_logging("test_logger", "DEBUG")
    assert logger.level == 10  # DEBUG level

def test_setup_api():
    # Skip if no API key in environment
    if not os.getenv('GOOGLE_API_KEY'):
        pytest.skip("GOOGLE_API_KEY not set in environment")
    
    client = setup_api()
    assert client is not None
    
def test_setup_api_missing_key(monkeypatch):
    # Remove API key from environment
    monkeypatch.delenv('GOOGLE_API_KEY', raising=False)
    
    with pytest.raises(ValueError) as excinfo:
        setup_api()
    assert "GOOGLE_API_KEY not found" in str(excinfo.value)
