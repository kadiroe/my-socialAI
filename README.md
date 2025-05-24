# SocialAI

An AI-powered tool for generating and fine-tuning question-answer pairs from web content using Gemini API and FLAN-T5. This project uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and is optimized for CPU usage.

## Features

- Web content extraction with subdomain crawling capabilities
- Automated QA pair generation using Google's Gemini API
- Memory-efficient model fine-tuning using LoRA
- CPU-optimized training with FLAN-T5-small
- Memory-efficient training (works with <2GB RAM)
- Configurable data processing pipeline
- Comprehensive logging and error handling

## Prerequisites

- Python 3.9+
- uv (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/socialAI.git
cd socialAI
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

4. Install dependencies using uv:
```bash
uv pip install -e .
```

## Configuration

1. Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

2. The `config/config.yaml` contains the following configurations:
```yaml
web_extractor:
  user_agent: "Mozilla/5.0..."
  min_line_length: 30
  crawling:
    max_depth: 2  # How deep to crawl
    max_pages_per_domain: 10  # Maximum pages to crawl per domain
    include_subdomains: true  # Whether to crawl subdomains
    timeout: 30  # Timeout in seconds for each request
    concurrent_requests: 3  # Number of concurrent requests

qa_generator:
  model_name: "gemini-2.5-flash-preview-04-17"
  num_default_pairs: 5

fine_tuner:
  base_model: "google/flan-t5-small"  # Changed to T5 for better CPU performance
  training:
    num_train_epochs: 5.0
    per_device_train_batch_size: 4
    learning_rate: 5.0e-5
    weight_decay: 0.05
    max_grad_norm: 1.0
    warmup_ratio: 0.1
    lr_scheduler_type: "cosine"
  lora:
    r: 8
    lora_alpha: 32
    lora_dropout: 0.05
    target_modules: ["q", "k", "v", "o"]  # T5 attention components
    bias: "none"
    task_type: "SEQ_2_SEQ_LM"
```

## Usage

1. Run the main script:
```bash
python main.py
```

The script will:
1. Extract content from specified URLs with subdomain crawling
2. Generate QA pairs using Gemini API
3. Fine-tune FLAN-T5-small model using LoRA
4. Save the fine-tuned model
5. Run inference with example questions

## Project Structure

```
socialAI/
├── config/          # Configuration files
├── src/
│   ├── data/       # Data processing and web extraction
│   ├── models/     # QA generation and fine-tuning
│   └── utils/      # Helper utilities
├── tests/          # Test files
└── main.py         # Main execution script
```

## Model Details

The project uses FLAN-T5-small as the base model with LoRA fine-tuning for several reasons:
- Better performance on CPU compared to decoder-only models
- Smaller memory footprint while maintaining good quality
- Efficient fine-tuning with LoRA adaptation
- Strong performance on question-answering tasks

## Limitations

- CPU-only training can be slower than GPU-based training
- Response generation quality depends on the quality of training data
- Limited by the context window of FLAN-T5-small (512 tokens)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
│   └── utils/         # Logging and API utilities
├── tests/             # Unit tests
├── config/            # Configuration files
├── finetuned_model/   # Saved model outputs
└── main.py           # Main application
```

## Model Details

- Base Model: TinyLlama-1.1B-Chat
- Fine-tuning Method: LoRA (Low-Rank Adaptation)
- Optimization: CPU-optimized with reduced memory footprint
- Training Features:
  - Gradient accumulation
  - Dynamic batch sizing
  - Memory-efficient training

## Requirements

All dependencies are managed through pyproject.toml and include:
- accelerate
- beautifulsoup4
- datasets
- google-genai
- peft (for LoRA)
- torch
- transformers
- python-dotenv

## Testing

The project uses pytest for testing. To run the tests:

1. Install test dependencies:
```bash
uv pip install -e ".[test]"
```

2. Run all tests:
```bash
pytest
```

3. Run specific test files:
```bash
pytest tests/test_web_extractor.py
pytest tests/test_qa_generator.py
pytest tests/test_finetuner.py
```

4. Run tests with coverage report:
```bash
pytest --cov=src --cov-report=html
```

The tests cover:
- Web content extraction
- Dataset processing
- QA pair generation
- Model fine-tuning
- Utility functions

### Test Structure:
- `conftest.py`: Shared test fixtures and configuration
- `test_web_extractor.py`: Tests for web scraping functionality
- `test_dataset_processor.py`: Tests for dataset handling
- `test_qa_generator.py`: Tests for QA pair generation
- `test_finetuner.py`: Tests for model fine-tuning
- `test_utils.py`: Tests for utility functions

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

Kadir Özer
Project Link: [https://github.com/kadiroe/my-socialAI.git]