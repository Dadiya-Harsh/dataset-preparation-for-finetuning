#!/usr/bin/env python3
"""
GitHub and Online Documentation Data Collector for Code Fine-tuning

This script helps collect and format code examples from GitHub repositories,
Kaggle notebooks, and online documentation for fine-tuning code models.

It uses Ollama to dynamically generate meaningful, task-oriented instructions for code examples.
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
        Generate a meaningful, task-oriented instruction for code using Ollama LLM.
        Processes the full code, splitting into chunks if necessary to handle large files.
        
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
            # Build comprehensive context
            context = []
            
            # Add repository context if available
            if repo and file_path:
                context.append(f"File: {file_path}")
                context.append(f"Repository: {repo.full_name}")
                
                if repo.description:
                    context.append(f"Repository description: {repo.description}")
                
                # Try to get repo topics for more context
                try:
                    topics = repo.get_topics()
                    if topics:
                        context.append(f"Repository topics: {', '.join(topics)}")
                except:
                    pass
                
                # Get project structure summary
                try:
                    dir_path = os.path.dirname(file_path)
                    dir_contents = repo.get_contents(dir_path)
                    related_files = [os.path.basename(item.path) for item in dir_contents if item.type == "file"]
                    if related_files:
                        context.append(f"Related files in directory: {', '.join(related_files[:10])}")
                        
                    # Try to get README content for additional context
                    try:
                        readme_content = None
                        for item in dir_contents:
                            if item.name.lower() == 'readme.md':
                                readme_content = item.decoded_content.decode('utf-8', errors='replace')
                                break
                        
                        if readme_content:
                            # Extract first few paragraphs from README
                            readme_intro = '\n'.join(readme_content.split('\n\n')[:3])
                            if readme_intro:
                                context.append(f"Project README excerpt: {readme_intro}")
                    except:
                        pass
                except:
                    pass
                    
            if metadata:
                for key, value in metadata.items():
                    if key not in ['source', 'language'] and value:
                        context.append(f"{key}: {value}")
            
            # Process full code by splitting into manageable chunks if needed
            MAX_CHUNK_SIZE = 4000  # Characters per chunk
            full_code_length = len(code)
            
            # If code is small enough, send it in one request
            if full_code_length <= MAX_CHUNK_SIZE:
                return self._generate_instruction_single(code, language, context)
            
            # For larger code, process in multiple stages
            else:
                print(f"Code is large ({full_code_length} chars), processing in multiple chunks...")
                return self._generate_instruction_multi_chunk(code, language, context)
        
        except Exception as e:
            print(f"Error generating instruction with Ollama: {str(e)}")
            print(traceback.format_exc())
            return self._generate_fallback_instruction(code, language, repo, file_path)
            
    def _generate_instruction_single(self, code, language, context):
        """Generate instruction with full code in a single request"""
        
        # Create prompt that emphasizes practical, project-focused instructions
        prompt = f"""You are an expert at creating high-quality code instruction-response pairs for fine-tuning LLMs.

Given the following {language} code, create a SPECIFIC, PRACTICAL instruction that would prompt someone to write this code.

The instruction should:
1. Describe a REALISTIC PROJECT or TASK that would require writing this code
2. Include specific requirements, functionality details and purpose
3. Mention tools, libraries, and approaches that should be used
4. Be focused on the PURPOSE of the code, not just its structure
5. NOT be generic like "write a Python function" or "create a class"
6. NOT contain actual function/class names from the original code

Context information:
{chr(10).join(context)}

Code:
```{language.lower()}
{code}
```

Your task: Write a specific, detailed project-oriented instruction that would lead someone to write this exact code.
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
            return self._generate_fallback_instruction(code, language, None, None)
            
        result = response.json()
        instruction = result.get("response", "").strip()
        
        # Fallback if instruction is too short
        if len(instruction) < 30:
            print("LLM generated instruction was too short, using fallback.")
            return self._generate_fallback_instruction(code, language, None, None)
            
        return instruction

    def _generate_instruction_multi_chunk(self, code, language, context):
        """Process large code files in multiple steps"""
        
        # Step 1: Split code into manageable chunks
        MAX_CHUNK_SIZE = 4000
        code_chunks = []
        total_length = len(code)
        
        # Create reasonably-sized chunks, trying to split at meaningful boundaries
        chunk_start = 0
        while chunk_start < total_length:
            chunk_end = min(chunk_start + MAX_CHUNK_SIZE, total_length)
            
            # Try to find a good breaking point (newline) if not at the end
            if chunk_end < total_length:
                # Look for a newline within the last 200 chars of the chunk
                for i in range(chunk_end, max(chunk_start, chunk_end - 200), -1):
                    if code[i] == '\n':
                        chunk_end = i + 1
                        break
            
            code_chunks.append(code[chunk_start:chunk_end])
            chunk_start = chunk_end
        
        print(f"Split code into {len(code_chunks)} chunks")
        
        # Step 2: First pass - analyze each chunk to extract key information 
        chunk_analyses = []
        
        for i, chunk in enumerate(code_chunks):
            analysis_prompt = f"""You are analyzing a chunk of {language} code (chunk {i+1} of {len(code_chunks)}).
Extract the key components, functionality, and purpose of this code chunk.
Be concise but specific. Focus on identifying:
1. Main classes/functions and their purposes
2. Libraries/frameworks used
3. Core functionality implemented
4. Data structures or algorithms used

Code chunk:
```{language.lower()}
{chunk}
```

Respond with ONLY a concise analysis (3-5 sentences maximum)."""

            try:
                response = requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.ollama_model,
                        "prompt": analysis_prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.3,
                            "max_tokens": 200
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    chunk_analyses.append(result.get("response", "").strip())
                else:
                    chunk_analyses.append(f"Chunk {i+1}: Analysis failed")
                    
            except Exception as e:
                print(f"Error analyzing chunk {i+1}: {str(e)}")
                chunk_analyses.append(f"Chunk {i+1}: Analysis error")
                
            # Be nice to the API
            time.sleep(1)
        
        # Step 3: Generate final instruction based on all chunk analyses
        final_prompt = f"""You are creating a specific coding project instruction based on analyzed code.

The code being analyzed is written in {language} and consists of multiple parts.
Here's what each part of the code contains:

{chr(10).join(chunk_analyses)}

Additional context:
{chr(10).join(context)}

Based on this comprehensive analysis, create a SPECIFIC, PRACTICAL project instruction that would prompt someone to write this exact code.

The instruction should:
1. Describe a REALISTIC PROJECT that would require writing this code
2. Include specific requirements, functionality details and purpose
3. Mention necessary tools, libraries, and frameworks
4. NOT be generic like "write a Python script" or "create a class"
5. NOT contain the actual class/function names from the original code
6. Focus on the overall PURPOSE and functionality

Respond with ONLY the project instruction, nothing else."""

        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.ollama_model,
                    "prompt": final_prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 500
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                instruction = result.get("response", "").strip()
                
                # Validate instruction quality
                if len(instruction) < 30:
                    print("Generated instruction was too short, using fallback.")
                    return self._generate_fallback_instruction(code, language, None, None)
                    
                return instruction
            else:
                print(f"Final instruction generation failed: {response.status_code}")
                return self._generate_fallback_instruction(code, language, None, None)
                
        except Exception as e:
            print(f"Error generating final instruction: {str(e)}")
            return self._generate_fallback_instruction(code, language, None, None)
                
    def _generate_fallback_instruction(self, code, language, repo=None, file_path=None):
        """
        Enhanced fallback instruction generator when LLM generation fails
        Uses code analysis to create more meaningful instructions
        """
        description = f"Implement a {language} project"
        
        # Extract function/class names
        if language == "Python":
            functions = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[:\(]', code)
            # Extract imports for libraries
            imports = re.findall(r'import\s+([a-zA-Z0-9_.,\s]+)', code)
            imports += re.findall(r'from\s+([a-zA-Z0-9_.]+)\s+import', code)
        elif language in ["JavaScript", "TypeScript"]:
            functions = re.findall(r'function\s+([a-zA-Z0-9_]+)\s*\(', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[{\(]', code)
            # Extract imports/requires
            imports = re.findall(r'import\s+.*?from\s+[\'"]([a-zA-Z0-9_./\\-]+)[\'"]', code)
            imports += re.findall(r'require\([\'"]([a-zA-Z0-9_./\\-]+)[\'"]\)', code)
        elif language == "Java":
            functions = re.findall(r'(?:public|private|protected|static|\s) +[\w\<\>\[\]]+\s+(\w+) *\([^\)]*\)', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[{\(]', code)
            # Extract imports
            imports = re.findall(r'import\s+([a-zA-Z0-9_.]+);', code)
        else:
            functions = []
            classes = []
            imports = []
            
        # Extract docstrings and comments for understanding purpose
        comments = []
        if language == "Python":
            # Extract docstrings (multi-line and single line)
            docstrings = re.findall(r'"""(.+?)"""', code, re.DOTALL)
            docstrings += re.findall(r"'''(.+?)'''", code, re.DOTALL)
            comments = re.findall(r'#\s*(.+)$', code, re.MULTILINE) + docstrings
        elif language in ["JavaScript", "TypeScript", "Java", "C++", "C", "PHP"]:
            comments = re.findall(r'//\s*(.+)$', code, re.MULTILINE)
            # Extract multiline comments
            multiline = re.findall(r'/\*(.+?)\*/', code, re.DOTALL)
            comments.extend(multiline)
            
        # Extract file description from top comments/docstrings
        file_purpose = None
        if comments:
            # Check for file header docstring/comment
            top_comments = [c.strip() for c in comments[:3]]
            for comment in top_comments:
                if len(comment) > 30 and "import" not in comment.lower() and "copyright" not in comment.lower():
                    file_purpose = comment
                    break
        
        # Build more detailed description based on available information
        project_type = "tool" if "cli" in code.lower() or "argparse" in code.lower() else "library"
        
        if repo and file_path:
            file_name = os.path.basename(file_path)
            description = f"Develop a {language} {project_type} named '{file_name}'"
            
            if repo.description:
                description += f" for {repo.description}"
        
        # Add main functionality description
        if file_purpose:
            description += f" that {file_purpose}"
        elif classes and functions:
            # Identify main classes and functions
            main_components = []
            if len(classes) > 0:
                class_list = ", ".join(classes[:3])
                if len(classes) > 3:
                    class_list += ", and others"
                main_components.append(f"implements the {class_list} class{'es' if len(classes) > 1 else ''}")
                
            if len(functions) > 0:
                func_list = ", ".join(functions[:3])
                if len(functions) > 3:
                    func_list += ", and others"
                main_components.append(f"provides {func_list} function{'s' if len(functions) > 1 else ''}")
                
            if main_components:
                description += f" that {' and '.join(main_components)}"
        
        # Add important libraries/frameworks
        if imports:
            # Clean up imports
            cleaned_imports = []
            for imp in imports:
                # Split multi-imports
                for part in re.split(r'[,\s]+', imp):
                    part = part.strip()
                    if part and not part.startswith('.') and len(part) > 1:
                        # Get the base package
                        base_pkg = part.split('.')[0]
                        if base_pkg not in cleaned_imports and not base_pkg.startswith('_'):
                            cleaned_imports.append(base_pkg)
            
            if cleaned_imports:
                top_libs = cleaned_imports[:5]
                lib_text = ", ".join(top_libs)
                if len(cleaned_imports) > 5:
                    lib_text += ", and other libraries"
                description += f". The solution should use {lib_text}"
                
        # Add purpose from comments if available
        if comments and not file_purpose:
            # Find a comment that seems to describe functionality
            for comment in comments:
                comment = comment.strip()
                if len(comment) > 25 and len(comment) < 150:
                    if any(word in comment.lower() for word in ['implement', 'create', 'provide', 'generate', 'class', 'function', 'tool']):
                        description += f". The code should {comment}"
                        break
        
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
    
    def _get_repo_contents(self, repo, path="", max_depth=3, current_depth=0):
        """
        Recursively get repository contents with depth limit
        
        Args:
            repo: GitHub repository object
            path: Current path to get contents for
            max_depth: Maximum directory depth to recurse into
            current_depth: Current recursion depth
        """
        contents = []
        
        if current_depth > max_depth:
            return contents
            
        try:
            # Get contents of the current path
            items = repo.get_contents(path)
            
            for item in items:
                if item.type == "dir":
                    # Skip common directories we want to avoid
                    if any(skip in item.path.lower() for skip in [
                        'test', 'docs', 'example', 'vendor', 'node_modules', 
                        '.git', 'build', 'dist', '__pycache__'
                    ]):
                        continue
                        
                    # Recursively get directory contents with depth limit
                    if current_depth < max_depth:
                        contents.extend(self._get_repo_contents(
                            repo, item.path, max_depth, current_depth + 1
                        ))
                else:
                    contents.append(item)
        except Exception as e:
            print(f"Error getting repo contents for path {path}: {str(e)}")
                    
        return contents
    
    def _is_code_file(self, path):
        """Check if a file is a supported code file"""
        return any(path.endswith(ext) for ext in self.lang_extensions.keys())
    
    def _process_github_file(self, repo, file):
        """Process a GitHub file and create an example"""
        try:
            # Skip files with unwanted names
            file_basename = os.path.basename(file.path).lower()
            if any(skip in file_basename for skip in [
                'test', 'setup.py', '__init__.py', 'config', 'utils', 'helper'
            ]) and file.size < 1000:  # Allow larger util files that might be substantive
                return None
                
            # Get file content
            content = file.decoded_content.decode('utf-8', errors='replace')
            
            # Skip if file is too small or suspicious size
            if len(content) < 200:
                return None
                
            # Extract extension and determine language
            _, ext = os.path.splitext(file.path)
            language = self.lang_extensions.get(ext)
            
            if not language:
                return None
                
            # Quality checks to ensure the file is worth processing
            lines = content.count('\n')
            if lines < 15:  # Skip very small files
                return None
                
            # Check for code density (code vs whitespace/comments)
            non_empty_lines = sum(1 for line in content.splitlines() if line.strip())
            if non_empty_lines < 10:
                return None
                
            # Generate instruction using Ollama LLM
            instruction = self.generate_instruction_with_ollama(
                code=content,
                language=language,
                repo=repo,
                file_path=file.path
            )
            
            # Only create examples with substantive instructions
            if instruction and len(instruction) > 30:
                return {
                    "prompt": f"### Instruction:\n{instruction}\n\n### Response:",
                    "response": content,
                    "metadata": {
                        "source": "github",
                        "repo": repo.full_name,
                        "path": file.path,
                        "language": language,
                        "stars": repo.stargazers_count,
                        "file_size": file.size,
                        "line_count": lines
                    }
                }
            else:
                return None
            
        except Exception as e:
            print(f"Error processing file {file.path}: {str(e)}")
            return None
            
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
                    print(f"Processing Kaggle notebook {i+1}/{len(notebook_links)}")
                    
                    try:
                        nb_response = requests.get(link, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                        })
                        
                        if nb_response.status_code != 200:
                            continue
                            
                        nb_soup = BeautifulSoup(nb_response.text, 'html.parser')
                        
                        # Try to extract notebook title/description for context
                        notebook_title = nb_soup.find('h1', class_='notebook-title')
                        title_text = notebook_title.get_text().strip() if notebook_title else ""
                        
                        # Extract code cells (this will need adaptation)
                        code_cells = nb_soup.find_all('div', class_='source-code')
                        
                        for cell in code_cells:
                            code = cell.get_text()
                            
                            # Skip small code cells
                            if len(code) < 200 or code.count('\n') < 10:
                                continue
                                
                            # Get surrounding text for context
                            prev_markdown = cell.find_previous('div', class_='markdown-cell')
                            context_text = prev_markdown.get_text().strip() if prev_markdown else ""
                            
                            # Generate instruction using Ollama
                            instruction = self.generate_instruction_with_ollama(
                                code=code,
                                language="Python",  # Assume Python for Kaggle notebooks
                                metadata={
                                    "source": "kaggle",
                                    "url": link,
                                    "keyword": keyword,
                                    "notebook_title": title_text,
                                    "context": context_text[:300] if context_text else ""
                                }
                            )
                            
                            if instruction and len(instruction) > 30:
                                example = {
                                    "prompt": f"### Instruction:\n{instruction}\n\n### Response:",
                                    "response": code,
                                    "metadata": {
                                        "source": "kaggle",
                                        "url": link,
                                        "keyword": keyword,
                                        "notebook_title": title_text
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
            "docs.python.org": "pre.python",
            "developer.mozilla.org": "pre.brush",
            "docs.microsoft.com": "pre code",
            "docs.aws.amazon.com": "pre.programlisting",
            "docs.docker.com": "pre code",
            "kubernetes.io": "pre.language-yaml, pre.language-bash"
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
                title = soup.title.string.strip() if soup.title else "Documentation Example"
                
                # Try to extract page context (heading and surrounding text)
                main_content = soup.find('main') or soup.find('article') or soup.find('div', class_='content')
                page_context = ""
                if main_content:
                    headings = main_content.find_all(['h1', 'h2', 'h3'])
                    if headings:
                        page_context = headings[0].get_text().strip()
                        # Get first paragraph after heading
                        next_p = headings[0].find_next('p')
                        if next_p:
                            page_context += " - " + next_p.get_text().strip()[:200]
                
                # Extract code blocks
                code_blocks = soup.select(selector)
                
                for i, block in enumerate(code_blocks):
                    code = block.get_text().strip()
                    
                    # Skip small code blocks
                    if len(code) < 200 or code.count('\n') < 10:
                        continue
                        
                    # Try to determine language based on classes or parent classes
                    language = self._detect_language_from_html(block)
                    if not language:
                        language = "Unknown"
                    
                    # Get surrounding text for context
                    prev_p = block.find_previous('p')
                    context_text = prev_p.get_text().strip()[:300] if prev_p else ""
                    
                    # Generate instruction using Ollama
                    instruction = self.generate_instruction_with_ollama(
                        code=code,
                        language=language,
                        metadata={
                            "source": "documentation",
                            "url": url,
                            "title": title,
                            "context": context_text,
                            "page_context": page_context
                        }
                    )
                    
                    if instruction and len(instruction) > 30:
                        example = {
                            "prompt": f"### Instruction:\n{instruction}\n\n### Response:",
                            "response": code,
                            "metadata": {
                                "source": "documentation",
                                "url": url,
                                "title": title,
                                "language": language,
                                "context": context_text,
                                "page_context": page_context
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
    parser.add_argument("--files-per-repo", type=int, default=50, 
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