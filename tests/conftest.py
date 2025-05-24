import pytest
import yaml
import os

@pytest.fixture
def test_config():
    return {
        'web_extractor': {
            'user_agent': 'Mozilla/5.0 (Test)',
            'min_line_length': 30
        },
        'qa_generator': {
            'model_name': 'gemini-2.5-flash-preview-04-17',
            'num_default_pairs': 2
        },
        'fine_tuner': {
            'base_model': 'google/flan-t5-small',
            'training': {
                'num_train_epochs': 1.0,  # Reduced for testing
                'per_device_train_batch_size': 4,
                'learning_rate': 5.0e-5,
                'weight_decay': 0.05,
                'max_grad_norm': 1.0,
                'warmup_ratio': 0.1,
                'lr_scheduler_type': 'cosine'
            },
            'lora': {
                'r': 8,
                'lora_alpha': 32,
                'lora_dropout': 0.05,
                'target_modules': ['q', 'k', 'v', 'o'],
                'bias': 'none',
                'task_type': 'SEQ_2_SEQ_LM'
            }
        }
    }

@pytest.fixture
def test_qa_pairs():
    return [
        {
            'question': 'What services does the employment agency offer?',
            'answer': 'The employment agency offers job placement, career counseling, and unemployment benefits.'
        },
        {
            'question': 'How do I register as unemployed?',
            'answer': 'You can register as unemployed online or in person at your local employment agency office.'
        }
    ]

@pytest.fixture
def mock_webpage_content():
    return """
    Welcome to the Employment Agency
    We offer comprehensive services for job seekers and employers.
    Our main services include:
    - Job placement
    - Career counseling
    - Unemployment benefits
    - Professional training
    Please visit our local office or contact us online for more information.
    """
