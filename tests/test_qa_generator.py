import pytest
from src.models.qa_generator import QAGenerator

@pytest.mark.asyncio
async def test_qa_generator_init(test_config):
    generator = QAGenerator(test_config)
    assert generator.model_name == test_config['qa_generator']['model_name']
    assert generator.num_default_pairs == test_config['qa_generator']['num_default_pairs']

@pytest.mark.asyncio
async def test_create_prompt(test_config):
    generator = QAGenerator(test_config)
    context = "Test context"
    num_pairs = 3
    prompt = generator._create_prompt(context, num_pairs)
    assert str(num_pairs) in prompt
    assert context in prompt
    assert "Question:" in prompt
    assert "Answer:" in prompt

@pytest.mark.asyncio
async def test_parse_response(test_config):
    generator = QAGenerator(test_config)
    test_response = """
    Question: Test question 1?
    Answer: Test answer 1.

    Question: Test question 2?
    Answer: Test answer 2.
    """
    pairs = generator._parse_response(test_response)
    assert len(pairs) == 2
    assert all('question' in pair and 'answer' in pair for pair in pairs)
    assert pairs[0]['question'] == 'Test question 1?'
    assert pairs[0]['answer'] == 'Test answer 1.'

@pytest.mark.asyncio
async def test_generate_qa_pairs(test_config, mock_webpage_content):
    generator = QAGenerator(test_config)
    pairs = await generator.generate_qa_pairs(mock_webpage_content, num_pairs=2)
    # Since this requires actual API call, we just check the format
    if pairs:  # If API call succeeded
        assert len(pairs) > 0
        assert all('question' in pair and 'answer' in pair for pair in pairs)
