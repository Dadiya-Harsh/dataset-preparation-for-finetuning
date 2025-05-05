#!/usr/bin/env python3
"""
GitHub and Online Documentation Data Collector for Code Fine-tuning

This script helps collect and format code examples from GitHub repositories,
Kaggle notebooks, and online documentation for fine-tuning code models.

It uses Ollama to dynamically generate instructions for code examples.
"""

import os
import re
import json
import time
import random
import argparse
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from github import Github
from github.GithubException import RateLimitExceededException
from tqdm import tqdm
import traceback

class DataCollector:
    def __init__(self, github_token=None, output_dir="dataset", 
                 ollama_url="http://localhost:11434", ollama_model="codellama"):
        self.github_token = github_token
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize GitHub client if token is provided
        self.gh = Github(github_token) if github_token else None
        
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
        self.jsonl_path = self.output_dir / "fine_tuning_dataset.jsonl"
        
    def generate_instruction_with_ollama(self, code, language, repo=None, file_path=None, metadata=None):
        """
        Generate an instruction for code using Ollama LLM.
        
        Args:
            code: The code content to generate an instruction for
            language: The programming language of the code
            repo: Optional GitHub repository object
            file_path: Optional file path in the repository
            metadata: Additional metadata for context
            
        Returns:
            Generated instruction or None if generation failed
        """
        try:
            # Create a context-rich prompt for the LLM
            context = []
            if repo and file_path:
                context.append(f"File: {file_path}")
                context.append(f"Repository: {repo.full_name}")
                context.append(f"Repository description: {repo.description}")
                
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
                return self._generate_fallback_instruction(code, language, repo, file_path)
                
            result = response.json()
            instruction = result.get("response", "").strip()
            
            # Fallback to rule-based if the LLM output is too short or empty
            if len(instruction) < 20:
                print("LLM generated instruction was too short, using fallback.")
                return self._generate_fallback_instruction(code, language, repo, file_path)
                
            return instruction
            
        except Exception as e:
            print(f"Error generating instruction with Ollama: {str(e)}")
            print(traceback.format_exc())
            return self._generate_fallback_instruction(code, language, repo, file_path)
    
    def _generate_fallback_instruction(self, code, language, repo=None, file_path=None):
        """Legacy rule-based instruction generator as fallback"""
        description = f"Implement {language} code"
        
        # Extract function/class names
        if language == "Python":
            functions = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[:\(]', code)
        elif language in ["JavaScript", "TypeScript"]:
            functions = re.findall(r'function\s+([a-zA-Z0-9_]+)\s*\(', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[{\(]', code)
        elif language == "Java":
            functions = re.findall(r'(?:public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\)', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[{\(]', code)
        else:
            functions = []
            classes = []
            
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
        if repo and file_path:
            file_name = os.path.basename(file_path)
            description = f"Implement {language} code for a file named '{file_name}' from the '{repo.name}' repository"
            
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
    
    def collect_from_github(self, query, max_repos=10, files_per_repo=5, min_stars=100):
        """
        Collect code examples from GitHub repositories based on search query.
        
        Args:
            query: Search query for repositories
            max_repos: Maximum number of repositories to process
            files_per_repo: Maximum number of files to collect per repository
            min_stars: Minimum stars for repositories to consider
        """
        if not self.gh:
            print("GitHub token not provided. Skipping GitHub collection.")
            return []
        
        print(f"Searching GitHub for: {query}")
        examples = []
        
        # Search for repositories
        try:
            repos = self.gh.search_repositories(
                query=f"{query} stars:>={min_stars}", 
                sort="stars",
                order="desc"
            )
            
            # Process repositories
            for i, repo in enumerate(repos[:max_repos]):
                if i >= max_repos:
                    break
                    
                print(f"Processing repository {i+1}/{max_repos}: {repo.full_name} ({repo.stargazers_count} stars)")
                
                try:
                    # Get repository content
                    contents = self._get_repo_contents(repo)
                    
                    # Filter code files
                    code_files = [c for c in contents if self._is_code_file(c.path)]
                    
                    # Sample files if there are too many
                    if len(code_files) > files_per_repo:
                        code_files = random.sample(code_files, files_per_repo)
                    
                    # Process each file
                    for file in tqdm(code_files, desc="Files"):
                        example = self._process_github_file(repo, file)
                        if example:
                            examples.append(example)
                            
                            # Write example to JSONL file
                            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                                f.write(json.dumps(example) + "\n")
                                
                except Exception as e:
                    print(f"Error processing repo {repo.full_name}: {str(e)}")
                    
                # Sleep to avoid rate limiting
                time.sleep(2)
                
        except RateLimitExceededException:
            print("GitHub API rate limit exceeded. Try again later or use a token with higher rate limits.")
        except Exception as e:
            print(f"Error during GitHub collection: {str(e)}")
            
        print(f"Collected {len(examples)} examples from GitHub")
        return examples
    
    def _get_repo_contents(self, repo, path=""):
        """Recursively get repository contents"""
        contents = []
        
        # Get contents of the current path
        items = repo.get_contents(path)
        
        for item in items:
            if item.type == "dir":
                # Recursively get directory contents
                contents.extend(self._get_repo_contents(repo, item.path))
            else:
                contents.append(item)
                
        return contents
    
    def _is_code_file(self, path):
        """Check if a file is a supported code file"""
        return any(path.endswith(ext) for ext in self.lang_extensions.keys())
    
    def _process_github_file(self, repo, file):
        """Process a GitHub file and create an example"""
        try:
            # Get file content
            content = file.decoded_content.decode('utf-8', errors='replace')
            
            # Skip if file is too large or too small
            if len(content) > 10000 or len(content) < 100:
                return None
                
            # Extract extension and determine language
            _, ext = os.path.splitext(file.path)
            language = self.lang_extensions.get(ext)
            
            if not language:
                return None
                
            # Generate instruction using Ollama LLM
            instruction = self.generate_instruction_with_ollama(
                code=content,
                language=language,
                repo=repo,
                file_path=file.path
            )
            
            return {
                "prompt": f"### Instruction:\n{instruction}\n\n### Response:",
                "response": content,
                "metadata": {
                    "source": "github",
                    "repo": repo.full_name,
                    "path": file.path,
                    "language": language,
                    "stars": repo.stargazers_count
                }
            }
            
        except Exception as e:
            print(f"Error processing file {file.path}: {str(e)}")
            return None
            
    def collect_from_kaggle(self, keywords, max_notebooks=20):
        """
        Collect code examples from Kaggle notebooks using web scraping.
        Note: This is a simplified implementation and might need adjustments.
        
        Args:
            keywords: List of search keywords
            max_notebooks: Maximum number of notebooks to process
        """
        examples = []
        print("Collecting from Kaggle...")
        
        for keyword in keywords:
            try:
                # Search for notebooks (simplified)
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
                    print(f"Processing Kaggle notebook {i+1}/{len(notebook_links)}")
                    
                    try:
                        nb_response = requests.get(link, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        
                        if nb_response.status_code != 200:
                            continue
                            
                        nb_soup = BeautifulSoup(nb_response.text, 'html.parser')
                        
                        # Extract code cells (this will need adaptation)
                        code_cells = nb_soup.find_all('div', class_='source-code')
                        
                        for cell in code_cells:
                            code = cell.get_text()
                            
                            # Skip small code cells
                            if len(code) < 50:
                                continue
                                
                            # Generate instruction using Ollama
                            instruction = self.generate_instruction_with_ollama(
                                code=code,
                                language="Python",  # Assume Python for Kaggle notebooks
                                metadata={
                                    "source": "kaggle",
                                    "url": link,
                                    "keyword": keyword
                                }
                            )
                            
                            example = {
                                "prompt": f"### Instruction:\n{instruction}\n\n### Response:",
                                "response": code,
                                "metadata": {
                                    "source": "kaggle",
                                    "url": link,
                                    "keyword": keyword
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
        
        for url in urls:
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
                            "language": language
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
        clean_path = self.output_dir / "clean_dataset.jsonl"
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
    parser = argparse.ArgumentParser(description="Collect code examples for fine-tuning")
    parser.add_argument("--github-token", type=str, help="GitHub API token")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--github-query", type=str, default="language:python machine learning", 
                        help="GitHub search query")
    parser.add_argument("--max-repos", type=int, default=10, help="Maximum repositories to process")
    parser.add_argument("--files-per-repo", type=int, default=5, 
                        help="Maximum files per repository")
    parser.add_argument("--kaggle-keywords", type=str, default="machine learning, data science",
                        help="Comma-separated Kaggle search keywords")
    parser.add_argument("--docs-urls", type=str, 
                        default="https://docs.python.org/3/tutorial/,https://pytorch.org/tutorials/",
                        help="Comma-separated documentation URLs")
    parser.add_argument("--min-stars", type=int, default=100, 
                        help="Minimum stars for GitHub repositories")
    parser.add_argument("--ollama-url", type=str, default="http://localhost:11434",
                        help="URL of the Ollama API server")
    parser.add_argument("--ollama-model", type=str, default="codellama",
                        help="Ollama model to use for instruction generation")
    
    args = parser.parse_args()
    
    # Create collector
    collector = DataCollector(
        github_token=args.github_token,
        output_dir=args.output_dir,
        ollama_url=args.ollama_url,
        ollama_model=args.ollama_model
    )
    
    # Test Ollama connection
    if not collector.test_ollama_connection():
        print("Warning: Could not connect to Ollama. Will use fallback instruction generation.")
        print("Make sure Ollama is running and the model is downloaded.")
        print(f"You can download the model with: ollama pull {args.ollama_model}")
    
    # Collect from GitHub
    if args.github_token:
        collector.collect_from_github(
            query=args.github_query,
            max_repos=args.max_repos,
            files_per_repo=args.files_per_repo,
            min_stars=args.min_stars
        )
    else:
        print("No GitHub token provided. Skipping GitHub collection.")
    
    # Collect from Kaggle
    collector.collect_from_kaggle(
        keywords=args.kaggle_keywords.split(","),
        max_notebooks=20
    )
    
    # Collect from documentation
    collector.collect_from_documentation(
        urls=args.docs_urls.split(",")
    )
    
    # Merge and deduplicate
    collector.merge_and_deduplicate()

if __name__ == "__main__":
    main()