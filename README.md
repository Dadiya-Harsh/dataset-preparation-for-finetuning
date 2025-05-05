# LLM Code Fine-Tuning Data Collection

**My journey to prepare high-quality datasets for LLM fine-tuning on code**

## Overview

This repository contains tools for collecting and formatting code examples from multiple sources to create high-quality instruction-following datasets for fine-tuning Large Language Models (LLMs) on code generation tasks.

The data collector helps you gather diverse coding examples from:
- GitHub repositories
- Kaggle notebooks
- Online documentation sites

Each code example is paired with an automatically generated instruction that would prompt someone to write that specific code, creating instruction-response pairs ideal for fine-tuning.

## Features

- **Multi-source Collection**: Gather code from diverse sources to improve model versatility
- **Automatic Instruction Generation**: Uses Ollama LLM to dynamically create natural instructions for each code snippet
- **Language Support**: Works with multiple programming languages (Python, JavaScript, TypeScript, Java, C++, C, Go, Rust, Ruby, PHP, Swift, Kotlin)
- **Quality Control**: Filtering by repository stars, code size, and automatic deduplication
- **Fallback Systems**: Rule-based instruction generation when LLM generation fails
- **JSONL Output**: Dataset directly ready for fine-tuning workflows

## Requirements

- Python 3.6+
- GitHub API token (for GitHub collection)
- Ollama server running locally or remotely with CodeLlama model (or similar code-focused model)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/llm-code-fine-tuning.git
cd llm-code-fine-tuning

# Install dependencies
pip install -r requirements.txt

# Install Ollama (if using locally)
# Follow instructions at https://ollama.ai/download

# Pull CodeLlama model for instruction generation
ollama pull codellama
```

## Usage

Basic usage:

```bash
python data_collector.py --github-token YOUR_GITHUB_TOKEN
```

Advanced options:

```bash
python data_collector.py \
  --github-token YOUR_GITHUB_TOKEN \
  --output-dir dataset \
  --github-query "language:python machine learning" \
  --max-repos 10 \
  --files-per-repo 5 \
  --kaggle-keywords "machine learning, data science" \
  --docs-urls "https://docs.python.org/3/tutorial/,https://pytorch.org/tutorials/" \
  --min-stars 100 \
  --ollama-url "http://localhost:11434" \
  --ollama-model "codellama"
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--github-token` | GitHub API token | None |
| `--output-dir` | Output directory for datasets | `dataset` |
| `--github-query` | GitHub search query | `language:python machine learning` |
| `--max-repos` | Maximum repositories to process | 10 |
| `--files-per-repo` | Maximum files per repository | 5 |
| `--kaggle-keywords` | Comma-separated Kaggle search keywords | `machine learning, data science` |
| `--docs-urls` | Comma-separated documentation URLs | `https://docs.python.org/3/tutorial/,https://pytorch.org/tutorials/` |
| `--min-stars` | Minimum stars for GitHub repositories | 100 |
| `--ollama-url` | URL of the Ollama API server | `http://localhost:11434` |
| `--ollama-model` | Ollama model to use | `codellama` |

## Output Format

The tool generates two JSONL files in the output directory:

1. `fine_tuning_dataset.jsonl` - Complete dataset with metadata
2. `clean_dataset.jsonl` - Clean version ready for training

Each entry follows this format:

```json
{
  "prompt": "### Instruction:\n{instruction}\n\n### Response:",
  "response": "{code}",
  "metadata": {
    "source": "github",
    "repo": "username/repo",
    "path": "path/to/file.py",
    "language": "Python",
    "stars": 1024
  }
}
```

## Example Generated Instructions

The tool uses Ollama to generate natural instructions like:

- "Create a Python function that processes GitHub repository contents recursively, returning a flat list of file content objects."
- "Implement a JavaScript function for detecting programming language from HTML code block classes."
- "Write a data collector class in Python that can gather code examples from Kaggle notebooks using web scraping."

## My Journey & Insights

Throughout this project, I've learned several important lessons about preparing datasets for fine-tuning code LLMs:

1. **Quality over quantity**: High-quality examples from well-maintained repositories produce better results than massive datasets with low-quality code
   
2. **Instruction diversity**: The system generates various instruction styles to help the model generalize better

3. **Balanced representation**: Collecting from multiple sources helps prevent bias toward specific coding styles or patterns

4. **Processing challenges**: Web scraping can be unpredictable, especially for dynamically loaded content on sites like Kaggle

5. **Next steps**: I'm exploring ways to add more sources and improve instruction quality

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Thanks to the Ollama project for making local LLM inference accessible
- GitHub for their comprehensive API
- Various documentation sites that make their content available