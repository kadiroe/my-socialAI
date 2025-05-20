# SocialAI

An AI-powered tool for generating and fine-tuning question-answer pairs from web content using Gemini API and transformer models.

## Features

- Web content extraction from URLs
- Automated QA pair generation using Google's Gemini API
- Local model fine-tuning with Hugging Face transformers
- Configurable data processing pipeline
- Comprehensive logging and error handling

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/socialAI.git
cd socialAI
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Configuration

1. Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

2. Adjust settings in `config/config.yaml` according to your needs:
```yaml
web_extractor:
  user_agent: "Mozilla/5.0..."
  min_line_length: 30

qa_generator:
  model_name: "gemini-pro"
  num_default_pairs: 5

fine_tuner:
  base_model: "facebook/opt-125m"
  training:
    num_epochs: 3
    batch_size: 4
```

## Usage

Run the main script:
```bash
python main.py
```

## Project Structure

```
socialAI/
├── src/
│   ├── data/          # Data processing modules
│   ├── models/        # AI models and fine-tuning
│   └── utils/         # Utility functions
├── tests/             # Unit tests
├── config/            # Configuration files
└── main.py           # Main application
```

## Testing

Run the test suite:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Google Generative AI
- Beautiful Soup 4
- Datasets

## Contact

Kadir Özer
Project Link: [https://github.com/kadiroe/my-socialAI.git]