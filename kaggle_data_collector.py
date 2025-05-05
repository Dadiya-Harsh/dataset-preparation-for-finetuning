#!/usr/bin/env python3
"""
Kaggle Notebook Data Collector for Code Fine-tuning

This script helps collect and format code examples from Kaggle notebooks
for fine-tuning code models.

It uses Ollama to dynamically generate instructions for code examples.
"""

import os
import re
import json
import time
import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import traceback

class KaggleCollector:
    def __init__(self, output_dir="kaggle_dataset", 
                 ollama_url="http://localhost:11434", ollama_model="codellama"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Ollama LLM configuration
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        
        # JSON Lines output file
        self.jsonl_path = self.output_dir / "kaggle_dataset.jsonl"
        
    def generate_instruction_with_ollama(self, code, metadata=None):
        """
        Generate an instruction for code using Ollama LLM.
        
        Args:
            code: The code content to generate an instruction for
            metadata: Additional metadata for context
            
        Returns:
            Generated instruction or None if generation failed
        """
        try:
            # Create a context-rich prompt for the LLM
            context = []
            
            if metadata:
                for key, value in metadata.items():
                    if key not in ['source', 'language'] and value:
                        context.append(f"{key}: {value}")
            
            # Truncate code if it's too large to avoid token limits
            code_preview = code[:5000] + "..." if len(code) > 5000 else code
            
            prompt = f"""You are an expert at creating high-quality code instruction-response pairs for fine-tuning LLMs.

Given the following Python code, write a clear, specific instruction that would prompt someone to write this exact code.
The instruction should be detailed enough that the code would be a perfect response to it.

Context:
{chr(10).join(context)}

Code:
```python
{code_preview}
```

Your task: Write a clear instruction (prompt) that would lead someone to write this code. 
Be specific about functionality, purpose, and requirements. Don't repeat the code itself in the instruction.
Respond with ONLY the instruction text, nothing else."""

            # Make API call to Ollama
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 500
                    }
                }
            )
            
            if response.status_code != 200:
                print(f"Ollama API error: {response.status_code} - {response.text}")
                return self._generate_fallback_instruction(code)
                
            result = response.json()
            instruction = result.get("response", "").strip()
            
            # Fallback to rule-based if the LLM output is too short or empty
            if len(instruction) < 20:
                print("LLM generated instruction was too short, using fallback.")
                return self._generate_fallback_instruction(code)
                
            return instruction
            
        except Exception as e:
            print(f"Error generating instruction with Ollama: {str(e)}")
            print(traceback.format_exc())
            return self._generate_fallback_instruction(code)
    
    def _generate_fallback_instruction(self, code):
        """Legacy rule-based instruction generator as fallback"""
        description = "Implement Python code"
        
        # Extract function/class names
        functions = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(', code)
        classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[:\(]', code)
            
        # Extract comments to understand purpose
        # Extract docstrings
        docstrings = re.findall(r'"""(.+?)"""', code, re.DOTALL)
        comments = re.findall(r'#\s*(.+)$', code, re.MULTILINE) + docstrings
            
        # Build description based on available information
        if classes:
            class_list = ", ".join(classes[:3])
            if len(classes) > 3:
                class_list += ", etc."
            description += f" that defines {class_list} class{'es' if len(classes) > 1 else ''}"
                
        if functions and not "that defines" in description:
            func_list = ", ".join(functions[:3])
            if len(functions) > 3:
                func_list += ", etc."
            description += f" that implements {func_list} function{'s' if len(functions) > 1 else ''}"
                
        # Add some context from comments if available
        if comments:
            # Get the first non-empty comment that's reasonably sized
            informative_comments = [c.strip() for c in comments if len(c.strip()) > 10 and len(c.strip()) < 100]
            if informative_comments:
                description += f". The code {informative_comments[0]}"
                
        return description
    
    def collect_from_kaggle(self, keywords, max_notebooks=20):
        """
        Collect code examples from Kaggle notebooks using web scraping.
        
        Args:
            keywords: List of search keywords
            max_notebooks: Maximum number of notebooks to process
        """
        examples = []
        print("Collecting from Kaggle...")
        
        for keyword in keywords:
            try:
                print(f"Searching Kaggle for: {keyword}")
                # Search for notebooks
                url = f"https://www.kaggle.com/code?searchQuery={keyword}"
                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code != 200:
                    print(f"Failed to search Kaggle for {keyword}: {response.status_code}")
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract notebook links (this will need to be adapted to Kaggle's actual HTML structure)
                notebook_links = []
                for a in soup.find_all('a', href=True):
                    if '/code/' in a['href'] and not 'searchQuery' in a['href']:
                        notebook_links.append(f"https://www.kaggle.com{a['href']}")
                
                notebook_links = list(set(notebook_links))[:max_notebooks]
                
                # Process each notebook
                for i, link in enumerate(notebook_links):
                    print(f"Processing Kaggle notebook {i+1}/{len(notebook_links)}: {link}")
                    
                    try:
                        nb_response = requests.get(link, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        
                        if nb_response.status_code != 200:
                            continue
                            
                        nb_soup = BeautifulSoup(nb_response.text, 'html.parser')
                        
                        # Extract title
                        title = nb_soup.title.string if nb_soup.title else "Kaggle Notebook"
                        
                        # Extract code cells (this will need adaptation)
                        code_cells = nb_soup.find_all('div', class_='source-code')
                        
                        print(f"Found {len(code_cells)} code cells in notebook")
                        
                        for cell_idx, cell in enumerate(code_cells):
                            code = cell.get_text()
                            
                            # Skip small code cells
                            if len(code) < 50:
                                continue
                                
                            # Generate instruction using Ollama
                            instruction = self.generate_instruction_with_ollama(
                                code=code,
                                metadata={
                                    "source": "kaggle",
                                    "url": link,
                                    "keyword": keyword,
                                    "title": title
                                }
                            )
                            
                            example = {
                                "prompt": f"### Instruction:\n{instruction}\n\n### Response:",
                                "response": code,
                                "metadata": {
                                    "source": "kaggle",
                                    "url": link,
                                    "keyword": keyword,
                                    "title": title,
                                    "cell_index": cell_idx
                                }
                            }
                            
                            examples.append(example)
                            
                            # Write example to JSONL file
                            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(example) + "\n")
                                
                    except Exception as e:
                        print(f"Error processing notebook {link}: {str(e)}")
                        
                    # Sleep to be nice to Kaggle
                    time.sleep(3)
                    
            except Exception as e:
                print(f"Error during Kaggle collection for {keyword}: {str(e)}")
                
        print(f"Collected {len(examples)} examples from Kaggle")
        return examples

    def test_ollama_connection(self):
        """Test connection to Ollama server"""
        try:
            # Simple test prompt
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": "Say hello",
                    "stream": False,
                    "options": {
                        "temperature": 0.7
                    }
                }
            )
            
            if response.status_code == 200:
                print(f"✅ Successfully connected to Ollama at {self.ollama_url}")
                print(f"Model: {self.ollama_model}")
                return True
            else:
                print(f"❌ Failed to connect to Ollama: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error connecting to Ollama: {str(e)}")
            return False
            
    def merge_and_deduplicate(self):
        """Merge and deduplicate collected examples"""
        # Read all examples
        examples = []
        if self.jsonl_path.exists():
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line))
                    
        # Remove duplicates based on response content
        seen_responses = set()
        unique_examples = []
        
        for example in examples:
            # Create a hash of the response to check for duplicates
            response_hash = hash(example['response'])
            
            if response_hash not in seen_responses:
                seen_responses.add(response_hash)
                unique_examples.append(example)
                
        # Write back the deduplicated examples
        with open(self.jsonl_path, 'w', encoding='utf-8') as f:
            for example in unique_examples:
                f.write(json.dumps(example) + '\n')
                
        print(f"Deduplicated dataset: {len(examples)} -> {len(unique_examples)} examples")
        
        # Also write a clean version for training
        clean_path = self.output_dir / "clean_kaggle_dataset.jsonl"
        with open(clean_path, 'w', encoding='utf-8') as f:
            for example in unique_examples:
                # Write only prompt and response fields
                clean_example = {
                    "prompt": example["prompt"],
                    "response": example["response"]
                }
                f.write(json.dumps(clean_example) + '\n')
                
        print(f"Clean dataset for training saved to {clean_path}")
        
        return unique_examples

def main():
    parser = argparse.ArgumentParser(description="Collect code examples from Kaggle for fine-tuning")
    parser.add_argument("--output-dir", type=str, default="kaggle_dataset", 
                        help="Output directory")
    parser.add_argument("--kaggle-keywords", type=str, default="machine learning, data science, python, pandas",
                        help="Comma-separated Kaggle search keywords")
    parser.add_argument("--max-notebooks", type=int, default=20,
                        help="Maximum number of notebooks to process per keyword")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="URL of the Ollama API server")
    parser.add_argument("--ollama-model", type=str, default="codellama",
                        help="Ollama model to use for instruction generation")
    
    args = parser.parse_args()
    
    # Create collector
    collector = KaggleCollector(
        output_dir=args.output_dir,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model
    )
    
    # Test Ollama connection
    if not collector.test_ollama_connection():
        print("Warning: Could not connect to Ollama. Will use fallback instruction generation.")
        print("Make sure Ollama is running and the model is downloaded.")
        print(f"You can download the model with: ollama pull {args.ollama_model}")
    
    # Collect from Kaggle
    collector.collect_from_kaggle(
        keywords=args.kaggle_keywords.split(","),
        max_notebooks=args.max_notebooks
    )
    
    # Merge and deduplicate
    collector.merge_and_deduplicate()

if __name__ == "__main__":
    main()