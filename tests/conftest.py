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
            'base_model': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'training': {
                'num_train_epochs': 1.0,
                'per_device_train_batch_size': 2,
                'learning_rate': 2.0e-4,
                'weight_decay': 0.01,
                'max_grad_norm': 0.3,
                'warmup_ratio': 0.03,
                'lr_scheduler_type': 'cosine',
                'dataloader_pin_memory': False
            },
            'lora': {
                'r': 8,
                'lora_alpha': 32,
                'lora_dropout': 0.1,
                'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                'bias': 'none',
                'task_type': 'CAUSAL_LM'
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
