#!/usr/bin/env python3
"""
Test script to collect data from a single GitHub repository
"""

import os
import json
import time
import argparse
from github import Github
from pathlib import Path
from tqdm import tqdm

def test_single_repo(github_token, repo_name, output_dir="test_dataset", files_limit=10):
    """
    Test data collection on a single GitHub repository.
    
    Args:
        github_token: GitHub API token
        repo_name: Full name of the repository (e.g., 'username/repo')
        output_dir: Directory to save the output
        files_limit: Maximum number of files to collect
    """
    if not github_token:
        print("GitHub token is required to access repositories")
        return
    
    # Create output directory
    output_dir = Path(f"data/{output_dir}")
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Initialize GitHub client
    gh = Github(github_token)
    
    # Language mapping
    lang_extensions = {
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
    
    # Output JSONL file
    jsonl_path = output_dir / "test_dataset.jsonl"
    
    # Get repository
    try:
        repo = gh.get_repo(repo_name)
        print(f"Successfully accessed repository: {repo.full_name} ({repo.stargazers_count} stars)")
        
        # Get repository contents recursively
        contents = get_repo_contents(repo)
        
        # Filter code files
        code_files = [c for c in contents if is_code_file(c.path, lang_extensions)]
        print(f"Found {len(code_files)} code files in repository")
        
        # Limit number of files if needed
        if len(code_files) > files_limit:
            print(f"Limiting to {files_limit} files")
            code_files = code_files[:files_limit]
        
        # Process each file
        examples = []
        for file in tqdm(code_files, desc="Processing files"):
            example = process_github_file(repo, file, lang_extensions)
            if example:
                examples.append(example)
                
                # Write example to JSONL file
                with open(jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(example) + "\n")
                    
        print(f"Successfully collected {len(examples)} examples from {repo.full_name}")
        print(f"Results saved to {jsonl_path}")
        
        # Also save a summary file with file paths
        summary_path = output_dir / "file_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Repository: {repo.full_name}\n")
            f.write(f"Total files processed: {len(code_files)}\n")
            f.write(f"Examples collected: {len(examples)}\n\n")
            f.write("Files processed:\n")
            for file in code_files:
                f.write(f"- {file.path}\n")
                
        print(f"Summary saved to {summary_path}")
        
        # Save a few examples in human-readable format
        examples_path = output_dir / "example_samples.md"
        with open(examples_path, "w", encoding="utf-8") as f:
            f.write(f"# Example Samples from {repo.full_name}\n\n")
            for i, example in enumerate(examples[:5]):  # Show first 5 examples
                f.write(f"## Example {i+1}\n\n")
                f.write(f"### Prompt\n\n```\n{example['prompt']}\n```\n\n")
                f.write(f"### Response\n\n```\n{example['response'][:500]}{'...' if len(example['response']) > 500 else ''}\n```\n\n")
                f.write("-" * 80 + "\n\n")
                
        print(f"Example samples saved to {examples_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

def get_repo_contents(repo, path=""):
    """Recursively get repository contents"""
    contents = []
    
    try:
        # Get contents of the current path
        items = repo.get_contents(path)
        
        for item in items:
            if item.type == "dir":
                # Recursively get directory contents, but avoid going too deep
                if path.count('/') < 3:  # Limit recursion depth for testing
                    contents.extend(get_repo_contents(repo, item.path))
            else:
                contents.append(item)
                
    except Exception as e:
        print(f"Error accessing {path}: {str(e)}")
    
    return contents

def is_code_file(path, lang_extensions):
    """Check if a file is a supported code file"""
    return any(path.endswith(ext) for ext in lang_extensions.keys())

def process_github_file(repo, file, lang_extensions):
    """Process a GitHub file and create an example"""
    try:
        # Get file content
        content = file.decoded_content.decode('utf-8', errors='replace')
        
        # Skip if file is too large or too small
        if len(content) > 10000 or len(content) < 100:
            return None
            
        # Extract extension and determine language
        _, ext = os.path.splitext(file.path)
        language = lang_extensions.get(ext)
        
        if not language:
            return None
            
        # Create a simple description for testing
        file_name = os.path.basename(file.path)
        description = f"Implement {language} code for a file named '{file_name}' from the '{repo.name}' repository"
        
        return {
            "prompt": f"### Instruction:\n{description}\n\n### Response:",
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

def main():
    parser = argparse.ArgumentParser(description="Test data collection on a single GitHub repository")
    parser.add_argument("--github-token", type=str, required=True, help="GitHub API token")
    parser.add_argument("--repo", type=str, required=True, help="Repository name (e.g., 'username/repo')")
    parser.add_argument("--output-dir", type=str, default="test_dataset", help="Output directory")
    parser.add_argument("--files-limit", type=int, default=10, help="Maximum number of files to collect")
    
    args = parser.parse_args()
    
    test_single_repo(
        github_token=args.github_token,
        repo_name=args.repo,
        output_dir=args.output_dir,
        files_limit=args.files_limit
    )

if __name__ == "__main__":
    main()