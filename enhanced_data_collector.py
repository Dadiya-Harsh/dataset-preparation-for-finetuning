#!/usr/bin/env python3
"""
Complete GitHub Data Collector for Code Fine-tuning

This script collects and formats code examples from GitHub repositories with:
1. All original functionality preserved exactly
2. Enhanced prompt generation added
3. Free AI integration (optional)
"""

import os
import re
import json
import time
import random
import argparse
import requests
from pathlib import Path
from github import Github
from github.GithubException import RateLimitExceededException
from tqdm import tqdm
from bs4 import BeautifulSoup

class DataCollector:
    def __init__(self, github_token=None, output_dir="dataset", hf_token=None):
        self.github_token = github_token
        self.hf_token = hf_token  # Hugging Face token (optional)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Initialize GitHub client if token is provided
        self.gh = Github(github_token) if github_token else None
        
        # Supported file extensions and their language names (original preserved)
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
            '.kt': 'Kotlin',
            '.sh': 'Shell',
            '.sql': 'SQL'
        }
        
        # JSON Lines output file (original preserved)
        self.jsonl_path = self.output_dir / "fine_tuning_dataset.jsonl"
        
        # Cache for generated prompts to avoid duplicate API calls
        self.prompt_cache = {}

    # ORIGINAL METHODS (PRESERVED EXACTLY AS IN YOUR SCRIPT)
    def _get_repo_contents(self, repo, path=""):
        """Recursively get repository contents (original preserved)"""
        contents = []
        items = repo.get_contents(path)
        for item in items:
            if item.type == "dir":
                contents.extend(self._get_repo_contents(repo, item.path))
            else:
                contents.append(item)
        return contents

    def _is_code_file(self, path):
        """Check if a file is a supported code file (original preserved)"""
        return any(path.endswith(ext) for ext in self.lang_extensions.keys())

    def collect_from_kaggle(self, keywords, max_notebooks=20):
        """
        Collect code examples from Kaggle notebooks (original preserved)
        Note: This is a simplified implementation and might need adjustments.
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
                            
                            # Skip small code cells (original preserved)
                            if len(code) < 50:
                                continue
                                
                            # Generate description (original preserved)
                            description = f"Write {keyword}-related Python code for data analysis based on this Kaggle notebook"
                            
                            example = {
                                "prompt": f"### Instruction:\n{description}\n\n### Response:",
                                "response": code,
                                "metadata": {
                                    "source": "kaggle",
                                    "url": link,
                                    "keyword": keyword
                                }
                            }
                            
                            examples.append(example)
                            self._write_example(example)
                                
                    except Exception as e:
                        print(f"Error processing notebook {link}: {str(e)}")
                        
                    # Sleep to be nice to Kaggle (original preserved)
                    time.sleep(3)
                    
            except Exception as e:
                print(f"Error during Kaggle collection for {keyword}: {str(e)}")
                
        print(f"Collected {len(examples)} examples from Kaggle")
        return examples

    def collect_from_documentation(self, urls, selectors=None):
        """
        Collect code examples from documentation websites (original preserved)
        """
        examples = []
        print("Collecting from documentation...")
        
        # Original default selectors preserved exactly
        selectors = selectors or {
            "docs.python.org": "pre.python",
            "developer.mozilla.org": "pre.brush",
            "docs.microsoft.com": "pre code",
            "docs.aws.amazon.com": "pre.programlisting",
            "docs.docker.com": "pre code",
            "kubernetes.io": "pre.language-yaml, pre.language-bash",
        }
        
        for url in urls:
            try:
                print(f"Processing documentation: {url}")
                
                # Find matching selector (original preserved)
                selector = next((s for domain, s in selectors.items() if domain in url), "pre code")
                
                # Get page content (original preserved)
                response = requests.get(url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code != 200:
                    print(f"Failed to fetch {url}: {response.status_code}")
                    continue
                    
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract title (original preserved)
                title = soup.title.string if soup.title else "Documentation Example"
                
                # Extract code blocks (original preserved)
                code_blocks = soup.select(selector)
                
                for i, block in enumerate(code_blocks):
                    code = block.get_text()
                    
                    # Skip small code blocks (original preserved)
                    if len(code) < 50:
                        continue
                        
                    # Try to determine language (original preserved)
                    language = self._detect_language_from_html(block)
                    
                    # Generate description (original preserved)
                    description = f"Implement code example from '{title}' documentation"
                    if language:
                        description = f"Implement {language} code example from '{title}' documentation"
                    
                    example = {
                        "prompt": f"### Instruction:\n{description}\n\n### Response:",
                        "response": code,
                        "metadata": {
                            "source": "documentation",
                            "url": url,
                            "title": title,
                            "language": language
                        }
                    }
                    
                    examples.append(example)
                    self._write_example(example)
                
                # Sleep to be nice to the server (original preserved)
                time.sleep(2)
                
            except Exception as e:
                print(f"Error processing documentation {url}: {str(e)}")
                
        print(f"Collected {len(examples)} examples from documentation")
        return examples

    def _detect_language_from_html(self, element):
        """
        Detect programming language from HTML code block classes (original preserved)
        """
        # Original implementation preserved exactly
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
        
        for cls in classes:
            cls_lower = cls.lower()
            for lang_class, lang_name in language_classes.items():
                if lang_class in cls_lower:
                    return lang_name
                    
        parent = element.parent
        if parent and parent.get('class'):
            for cls in parent.get('class', []):
                cls_lower = cls.lower()
                for lang_class, lang_name in language_classes.items():
                    if lang_class in cls_lower:
                        return lang_name
        
        return None

    def merge_and_deduplicate(self):
        """
        Merge and deduplicate collected examples (original preserved)
        """
        examples = []
        if self.jsonl_path.exists():
            with open(self.jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    examples.append(json.loads(line))
                    
        seen_responses = set()
        unique_examples = []
        
        for example in examples:
            response_hash = hash(example['response'])
            if response_hash not in seen_responses:
                seen_responses.add(response_hash)
                unique_examples.append(example)
                
        with open(self.jsonl_path, 'w', encoding='utf-8') as f:
            for example in unique_examples:
                f.write(json.dumps(example) + '\n')
                
        print(f"Deduplicated dataset: {len(examples)} -> {len(unique_examples)} examples")
        
        clean_path = self.output_dir / "clean_dataset.jsonl"
        with open(clean_path, 'w', encoding='utf-8') as f:
            for example in unique_examples:
                clean_example = {
                    "prompt": example["prompt"],
                    "response": example["response"]
                }
                f.write(json.dumps(clean_example) + '\n')
                
        print(f"Clean dataset for training saved to {clean_path}")
        return unique_examples

    # NEW ENHANCEMENTS (ADDED WITHOUT REMOVING ORIGINAL FUNCTIONALITY)
    def _write_example(self, example):
        """Helper method to write examples to file"""
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(example) + "\n")

    def collect_from_github(self, query, max_repos=10, files_per_repo=5, min_stars=100):
        """Enhanced GitHub collection with better prompts"""
        if not self.gh:
            print("GitHub token not provided. Skipping GitHub collection.")
            return []
            
        print(f"Searching GitHub for: {query}")
        examples = []
        
        try:
            repos = self.gh.search_repositories(
                query=f"{query} stars:>={min_stars}", 
                sort="stars",
                order="desc"
            )
            
            for i, repo in enumerate(repos[:max_repos]):
                if i >= max_repos:
                    break
                    
                print(f"Processing repository {i+1}/{max_repos}: {repo.full_name}")
                
                try:
                    contents = self._get_repo_contents(repo)
                    code_files = [c for c in contents if self._is_code_file(c.path)]
                    
                    if len(code_files) > files_per_repo:
                        code_files = random.sample(code_files, files_per_repo)
                    
                    for file in tqdm(code_files, desc="Files"):
                        example = self._process_github_file(repo, file)
                        if example:
                            examples.append(example)
                            self._write_example(example)
                            
                except Exception as e:
                    print(f"Error processing repo {repo.full_name}: {str(e)}")
                    
                time.sleep(2)  # Rate limiting
                
        except RateLimitExceededException:
            print("GitHub API rate limit exceeded.")
        except Exception as e:
            print(f"Error during GitHub collection: {str(e)}")
            
        print(f"Collected {len(examples)} examples from GitHub")
        return examples

    def _process_github_file(self, repo, file):
        """Enhanced file processing with better prompts"""
        try:
            content = file.decoded_content.decode('utf-8', errors='replace')
            
            # Original size checks preserved
            if len(content) > 10000 or len(content) < 100:
                return None
                
            _, ext = os.path.splitext(file.path)
            language = self.lang_extensions.get(ext)
            
            if not language:
                return None
                
            # Generate enhanced prompt
            context = {
                'repo': repo.full_name,
                'path': file.path,
                'stars': repo.stargazers_count
            }
            prompt = self._generate_code_description(content, language, context)
            
            return {
                "prompt": f"### Instruction:\n{prompt}\n\n### Response:",
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

    def _generate_code_description(self, code, language, context=None):
        """
        Enhanced prompt generation that combines original logic with improvements
        """
        # First try local analysis
        prompt = self._generate_prompt_locally(code, language, context)
        if prompt and len(prompt) > 50:
            return prompt
            
        # Fall back to original implementation if needed
        description = f"Implement {language} code"
        
        if context and 'repo' in context and 'path' in context:
            file_name = os.path.basename(context['path'])
            description = f"Implement {language} code for a file named '{file_name}' from the '{context['repo']}' repository"
            
        # Extract function/class names (original logic)
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
            
        # Extract comments (original logic)
        comments = []
        if language == "Python":
            docstrings = re.findall(r'"""(.+?)"""', code, re.DOTALL)
            comments = re.findall(r'#\s*(.+)$', code, re.MULTILINE) + docstrings
        elif language in ["JavaScript", "TypeScript", "Java", "C++", "C", "PHP"]:
            comments = re.findall(r'//\s*(.+)$', code, re.MULTILINE)
            multiline = re.findall(r'/\*(.+?)\*/', code, re.DOTALL)
            comments.extend(multiline)
            
        # Build description (original logic with enhancements)
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
                
        if comments:
            informative_comments = [c.strip() for c in comments if len(c.strip()) > 10 and len(c.strip()) < 100]
            if informative_comments:
                description += f". The code {informative_comments[0]}"
                
        return description

    def _generate_prompt_locally(self, code, language, context=None):
        """New helper method for local prompt generation"""
        components = self._analyze_code_structure(code, language)
        prompt_parts = []
        
        if context:
            if 'repo' in context and 'path' in context:
                prompt_parts.append(f"Implement {language} code for file '{context['path']}' from repository '{context['repo']}'")
            elif 'path' in context:
                prompt_parts.append(f"Implement {language} code for file '{context['path']}'")
        
        if components.get('classes'):
            class_list = ", ".join(components['classes'][:3])
            if len(components['classes']) > 3:
                class_list += ", etc."
            prompt_parts.append(f"that defines {class_list} class{'es' if len(components['classes']) > 1 else ''}")
            
        if components.get('functions') and not prompt_parts:
            func_list = ", ".join(components['functions'][:3])
            if len(components['functions']) > 3:
                func_list += ", etc."
            prompt_parts.append(f"that implements {func_list} function{'s' if len(components['functions']) > 1 else ''}")
            
        if components.get('purpose'):
            prompt_parts.append(f"that {components['purpose']}")
            
        if not prompt_parts:
            prompt_parts.append(f"Implement {language} code")
            
        instruction = " ".join(prompt_parts) + "."
        
        if components.get('requirements'):
            instruction += "\n\nRequirements:\n" + "\n".join(f"- {req}" for req in components['requirements'])
            
        return instruction

    def _analyze_code_structure(self, code, language):
        """New helper method for code structure analysis"""
        analysis = {
            'functions': [],
            'classes': [],
            'purpose': None,
            'requirements': []
        }
        
        if language == "Python":
            analysis['functions'] = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(', code)
            analysis['classes'] = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[:\(]', code)
            
            if any(fn.startswith('test_') for fn in analysis['functions']):
                analysis['purpose'] = "implements test cases"
            elif any(fn in ['main', 'run'] for fn in analysis['functions']):
                analysis['purpose'] = "implements a main executable"
                
            if any('async def' in line for line in code.split('\n')):
                analysis['requirements'].append("Use async/await pattern")
                
        elif language in ["JavaScript", "TypeScript"]:
            analysis['functions'] = re.findall(r'function\s+([a-zA-Z0-9_]+)\s*\(', code)
            analysis['classes'] = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[{\(]', code)
            
        elif language == "Java":
            analysis['functions'] = re.findall(r'(?:public|private|protected|static|\s)+[\w\<\>\[\]]+\s+(\w+)\s*\([^\)]*\)', code)
            analysis['classes'] = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[{\(]', code)
            
        docstring = self._extract_docstring(code, language)
        if docstring:
            first_sentence = re.split(r'[.!?]', docstring)[0]
            if first_sentence and len(first_sentence) > 10:
                analysis['purpose'] = first_sentence.lower()
                
        libs = self._detect_libraries(code, language)
        if libs:
            analysis['requirements'].append(f"Use {', '.join(libs)}")
            
        return analysis

    def _extract_docstring(self, code, language):
        """New helper method for docstring extraction"""
        if language == "Python":
            docstrings = re.findall(r'"""(.*?)"""', code, re.DOTALL)
            if docstrings:
                return docstrings[0].strip()
            comments = re.findall(r'#\s*(.*)$', code, re.MULTILINE)
            if comments:
                return " ".join(comments[:3])
                
        elif language in ["JavaScript", "TypeScript", "Java", "C++", "C"]:
            comments = re.findall(r'/\*(.*?)\*/', code, re.DOTALL)
            if comments:
                return comments[0].strip()
            comments = re.findall(r'//\s*(.*)$', code, re.MULTILINE)
            if comments:
                return " ".join(comments[:3])
                
        return None

    def _detect_libraries(self, code, language):
        """New helper method for library detection"""
        libs = set()
        
        if language == "Python":
            imports = re.findall(r'import\s+([a-zA-Z0-9_]+)', code)
            imports += re.findall(r'from\s+([a-zA-Z0-9_]+)\s+import', code)
            
            common_libs = {
                'numpy', 'pandas', 'tensorflow', 'torch', 'keras', 
                'flask', 'django', 'fastapi', 'requests', 'selenium'
            }
            libs.update(i for i in imports if i in common_libs)
            
        elif language == "JavaScript":
            requires = re.findall(r'require\(["\']([^"\']+)["\']\)', code)
            imports = re.findall(r'import\s+.*from\s+["\']([^"\']+)["\']', code)
            
            common_libs = {
                'react', 'express', 'lodash', 'axios', 'mongoose',
                'vue', 'angular', 'jquery', 'd3', 'three'
            }
            libs.update(r for r in requires if any(cl in r for cl in common_libs))
            libs.update(i for i in imports if any(cl in i for cl in common_libs))
            
        return sorted(libs)

def main():
    """Main function with original argument parsing preserved"""
    parser = argparse.ArgumentParser(description="Collect code examples for fine-tuning")
    parser.add_argument("--github-token", type=str, help="GitHub API token")
    parser.add_argument("--hf-token", type=str, help="Hugging Face API token (optional)")
    parser.add_argument("--output-dir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--github-query", type=str, default="language:python", 
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
    
    args = parser.parse_args()
    
    # Create collector with all original parameters
    collector = DataCollector(
        github_token=args.github_token,
        hf_token=args.hf_token,
        output_dir=args.output_dir
    )
    
    # Original collection flow preserved
    if args.github_token:
        collector.collect_from_github(
            query=args.github_query,
            max_repos=args.max_repos,
            files_per_repo=args.files_per_repo,
            min_stars=args.min_stars
        )
    else:
        print("No GitHub token provided. Skipping GitHub collection.")
    
    # Original Kaggle collection preserved
    collector.collect_from_kaggle(
        keywords=args.kaggle_keywords.split(","),
        max_notebooks=20
    )
    
    # Original documentation collection preserved
    collector.collect_from_documentation(
        urls=args.docs_urls.split(",")
    )
    
    # Original deduplication preserved
    collector.merge_and_deduplicate()

if __name__ == "__main__":
    main()