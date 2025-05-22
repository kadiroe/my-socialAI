import pytest
from src.data.web_extractor import WebExtractor

@pytest.mark.asyncio
async def test_web_extractor_init(test_config):
    extractor = WebExtractor(test_config)
    assert extractor.headers['User-Agent'] == test_config['web_extractor']['user_agent']
    assert extractor.min_line_length == test_config['web_extractor']['min_line_length']

@pytest.mark.asyncio
async def test_web_extractor_clean_content(test_config):
    extractor = WebExtractor(test_config)
    test_content = "Short line\nThis is a longer line that should be kept\nAnother short"
    cleaned = extractor._clean_content(test_content)
    assert len(cleaned.split('\n')) == 1
    assert "longer line" in cleaned

@pytest.mark.asyncio
async def test_web_extractor_validate_url(test_config):
    extractor = WebExtractor(test_config)
    assert extractor._is_valid_url("https://www.example.com") is True
    assert extractor._is_valid_url("not_a_url") is False

@pytest.mark.asyncio
async def test_get_webpage_content_invalid_url(test_config):
    extractor = WebExtractor(test_config)
    content = await extractor.get_webpage_content("not_a_url")
    assert content is None
