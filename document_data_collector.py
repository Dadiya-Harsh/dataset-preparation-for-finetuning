#!/usr/bin/env python3
"""
Documentation Website Data Collector for Code Fine-tuning

This script helps collect and format code examples from documentation websites
for fine-tuning code models.

It uses Ollama to dynamically generate instructions for code examples.
"""

import re
import json
import time
import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import traceback

class DocsCollector:
    def __init__(self, output_dir="docs_dataset", 
                 ollama_url="http://localhost:11434", ollama_model="codellama"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Ollama LLM configuration
        self.ollama_url = ollama_url
        self.ollama_model = ollama_model
        
        # Supported file extensions and their language names
        self.lang_extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript', 
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.go': 'Go',
            '.rs': 'Rust',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin'
        }
        
        # JSON Lines output file
        self.jsonl_path = self.output_dir / "docs_dataset.jsonl"
        
    def generate_instruction_with_ollama(self, code, language="Unknown", metadata=None):
        """
        Generate an instruction for code using Ollama LLM.
        
        Args:
            code: The code content to generate an instruction for
            language: The programming language of the code
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

Given the following {language} code, write a clear, specific instruction that would prompt someone to write this exact code.
The instruction should be detailed enough that the code would be a perfect response to it.

Context:
{chr(10).join(context)}

Code:
```{language.lower()}
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
                return self._generate_fallback_instruction(code, language)
                
            result = response.json()
            instruction = result.get("response", "").strip()
            
            # Fallback to rule-based if the LLM output is too short or empty
            if len(instruction) < 20:
                print("LLM generated instruction was too short, using fallback.")
                return self._generate_fallback_instruction(code, language)
                
            return instruction
            
        except Exception as e:
            print(f"Error generating instruction with Ollama: {str(e)}")
            print(traceback.format_exc())
            return self._generate_fallback_instruction(code, language)
    
    def _generate_fallback_instruction(self, code, language="Unknown"):
        """Legacy rule-based instruction generator as fallback"""
        description = f"Implement {language} code"
        
        # Extract function/class names based on language
        functions = []
        classes = []
        
        if language == "Python":
            functions = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[:\(]', code)
        elif language in ["JavaScript", "TypeScript"]:
            functions = re.findall(r'function\s+([a-zA-Z0-9_]+)\s*\(', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[{\(]', code)
        elif language == "Java":
            functions = re.findall(r'(?:public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\)', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[{\(]', code)
            
        # Extract comments to understand purpose
        comments = []
        if language == "Python":
            # Extract docstrings
            docstrings = re.findall(r'"""(.+?)"""', code, re.DOTALL)
            comments = re.findall(r'#\s*(.+)$', code, re.MULTILINE) + docstrings
        elif language in ["JavaScript", "TypeScript", "Java", "C++", "C", "PHP"]:
            comments = re.findall(r'//\s*(.+)$', code, re.MULTILINE)
            # Extract multiline comments
            multiline = re.findall(r'/\*(.+?)\*/', code, re.DOTALL)
            comments.extend(multiline)
            
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
    
    def collect_from_documentation(self, urls, selectors=None):
        """
        Collect code examples from documentation websites.
        
        Args:
            urls: List of documentation URLs to scrape
            selectors: Dictionary mapping URLs to CSS selectors for code blocks
        """
        examples = []
        print("Collecting from documentation...")
        
        selectors = selectors or {
            # Default selectors for common documentation sites
            "docs.python.org": "pre.python",
            "developer.mozilla.org": "pre.brush",
            "docs.microsoft.com": "pre code",
            "docs.aws.amazon.com": "pre.programlisting",
            "docs.docker.com": "pre code",
            "kubernetes.io": "pre.language-yaml, pre.language-bash",
            # Add more as needed
        }
        
        for url in tqdm(urls, desc="Documentation URLs"):
            try:
                print(f"Processing documentation: {url}")
                
                # Find matching selector
                selector = next((s for domain, s in selectors.items() if domain in url), "pre code")
                
                # Get page content
                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code != 200:
                    print(f"Failed to fetch {url}: {response.status_code}")
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title
                title = soup.title.string if soup.title else "Documentation Example"
                
                # Extract code blocks
                code_blocks = soup.select(selector)
                
                print(f"Found {len(code_blocks)} code blocks in {url}")
                
                for i, block in enumerate(code_blocks):
                    code = block.get_text()
                    
                    # Skip small code blocks
                    if len(code) < 50:
                        continue
                        
                    # Try to determine language based on classes or parent classes
                    language = self._detect_language_from_html(block)
                    
                    # Generate instruction using Ollama
                    instruction = self.generate_instruction_with_ollama(
                        code=code,
                        language=language if language else "Unknown",
                        metadata={
                            "source": "documentation",
                            "url": url,
                            "title": title
                        }
                    )
                    
                    example = {
                        "prompt": f"### Instruction:\n{instruction}\n\n### Response:",
                        "response": code,
                        "metadata": {
                            "source": "documentation",
                            "url": url,
                            "title": title,
                            "language": language,
                            "block_index": i
                        }
                    }
                    
                    examples.append(example)
                    
                    # Write example to JSONL file
                    with open(self.jsonl_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(example) + "\n")
                
                # Sleep to be nice to the server
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing documentation {url}: {str(e)}")
                print(traceback.format_exc())
                
        print(f"Collected {len(examples)} examples from documentation")
        return examples
    
    def _detect_language_from_html(self, element):
        """Detect programming language from HTML code block classes"""
        # Check element classes
        classes = element.get('class', [])
        
        language_classes = {
            'python': 'Python',
            'javascript': 'JavaScript',
            'typescript': 'TypeScript',
            'java': 'Java',
            'cpp': 'C++',
            'c': 'C',
            'csharp': 'C#',
            'go': 'Go',
            'rust': 'Rust',
            'ruby': 'Ruby',
            'php': 'PHP',
            'swift': 'Swift',
            'kotlin': 'Kotlin',
            'scala': 'Scala',
            'bash': 'Bash',
            'sh': 'Shell',
            'sql': 'SQL',
            'yaml': 'YAML',
            'json': 'JSON',
            'html': 'HTML',
            'css': 'CSS'
        }
        
        # Check for language classes
        for cls in classes:
            cls_lower = cls.lower()
            for lang_class, lang_name in language_classes.items():
                if lang_class in cls_lower:
                    return lang_name
                    
        # Check parent classes
        parent = element.parent
        if parent and parent.get('class'):
            for cls in parent.get('class', []):
                cls_lower = cls.lower()
                for lang_class, lang_name in language_classes.items():
                    if lang_class in cls_lower:
                        return lang_name
        
        return None

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
        clean_path = self.output_dir / "clean_docs_dataset.jsonl"
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

    def validate_dataset(self):
        """Validate the collected dataset for quality and consistency"""
        examples = []
        if not self.jsonl_path.exists():
            print("No dataset found for validation")
            return False
            
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        
        issues = []
        for i, example in enumerate(examples):
            # Check for required fields
            if not all(key in example for key in ['prompt', 'response', 'metadata']):
                issues.append(f"Example {i}: Missing required fields")
                continue
                
            # Check prompt quality
            if len(example['prompt']) < 20:
                issues.append(f"Example {i}: Prompt too short")
                
            # Check response quality
            if len(example['response']) < 50:
                issues.append(f"Example {i}: Response too short")
                
            # Check metadata
            if not all(key in example['metadata'] for key in ['source', 'url', 'title', 'language']):
                issues.append(f"Example {i}: Incomplete metadata")
                
        if issues:
            print("Dataset validation issues found:")
            for issue in issues:
                print(f"- {issue}")
            return False
        else:
            print("Dataset validation passed")
            return True

    def split_dataset(self, train_ratio=0.8, val_ratio=0.1):
        """Split dataset into train, validation, and test sets"""
        examples = []
        if not self.jsonl_path.exists():
            print("No dataset found for splitting")
            return
            
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
                
        if not examples:
            print("Empty dataset")
            return
            
        # Shuffle examples
        import random
        random.shuffle(examples)
        
        total = len(examples)
        train_size = int(total * train_ratio)
        val_size = int(total * val_ratio)
        
        train_examples = examples[:train_size]
        val_examples = examples[train_size:train_size + val_size]
        test_examples = examples[train_size + val_size:]
        
        # Save splits
        splits = {
            'train': (train_examples, self.output_dir / 'train.jsonl'),
            'val': (val_examples, self.output_dir / 'val.jsonl'),
            'test': (test_examples, self.output_dir / 'test.jsonl')
        }
        
        for split_name, (split_examples, split_path) in splits.items():
            with open(split_path, 'w', encoding='utf-8') as f:
                for example in split_examples:
                    f.write(json.dumps(example) + '\n')
            print(f"Saved {split_name} split with {len(split_examples)} examples to {split_path}")

def add_custom_selectors(selectors, custom_selectors_list):
    """Add custom selectors from command line to the selectors dictionary"""
    if not custom_selectors_list:
        return selectors
        
    for custom_selector in custom_selectors_list:
        try:
            domain, selector = custom_selector.split('=', 1)
            selectors[domain.strip()] = selector.strip()
        except ValueError:
            print(f"Warning: Ignoring invalid custom selector format: {custom_selector}")
            print("Custom selectors should be in format: domain=selector")
            
    return selectors

def main():
    parser = argparse.ArgumentParser(description="Collect code examples from documentation websites for fine-tuning")
    parser.add_argument("--output-dir", type=str, default="docs_dataset", 
                        help="Output directory")
    parser.add_argument("--docs-urls", type=str, 
                        default="https://docs.python.org/3/tutorial/,https://pytorch.org/tutorials/",
                        help="Comma-separated documentation URLs")
    parser.add_argument("--custom-selectors", type=str, default=None,
                        help="Comma-separated list of custom selectors in format domain=selector")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="URL of the Ollama API server")
    parser.add_argument("--ollama-model", type=str, default="codellama",
                        help="Ollama model to use for instruction generation")
    
    args = parser.parse_args()
    
    # Create collector
    collector = DocsCollector(
        output_dir=args.output_dir,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model
    )
    
    # Test Ollama connection
    if not collector.test_ollama_connection():
        print("Exiting due to Ollama connection failure")
        return
        
    # Parse URLs
    urls = args.docs_urls.split(',')
    
    # Parse custom selectors
    selectors = {
        "docs.python.org": "pre.python",
        "developer.mozilla.org": "pre.brush",
        "docs.microsoft.com": "pre code",
        "docs.aws.amazon.com": "pre.programlisting",
        "docs.docker.com": "pre code",
        "kubernetes.io": "pre.language-yaml, pre.language-bash"
    }
    
    if args.custom_selectors:
        custom_selectors_list = args.custom_selectors.split(',')
        selectors = add_custom_selectors(selectors, custom_selectors_list)
    
    # Collect examples
    collector.collect_from_documentation(urls, selectors)
    
    # Merge and deduplicate
    collector.merge_and_deduplicate()
    
    # Validate dataset
    if collector.validate_dataset():
        # Split dataset
        collector.split_dataset()
    
if __name__ == "__main__":
    main()