#!/usr/bin/env python3
"""
Complete Pipeline Script for DeepSeek-Coder Fine-tuning

This script coordinates the entire fine-tuning pipeline:
1. Collects code examples using data_collector.py
2. Generates quality instructions using code_instruction_generator.py
3. Fine-tunes DeepSeek-Coder on Ollama using deepseek_finetuner.py
"""

import os
import argparse
import subprocess
from pathlib import Path
import sys
import json

def check_script_exists(script_path):
    """Check if a script exists and is executable"""
    path = Path(script_path)
    return path.exists() and os.access(path, os.X_OK)

def make_executable(script_path):
    """Make a script executable"""
    os.chmod(script_path, 0o755)

def run_data_collection(args):
    """Run the data collection step"""
    print("\n=== Step 1: Data Collection ===")
    
    # Ensure data_collector.py exists and is executable
    collector_script = Path("data_collector.py")
    if not check_script_exists(collector_script):
        print(f"Making {collector_script} executable...")
        make_executable(collector_script)
    
    # Build command
    cmd = [
        "./data_collector_base.py",
        "--output-dir", args.output_dir
    ]
    
    if args.github_token:
        cmd.extend(["--github-token", args.github_token])
        
    if args.github_query:
        cmd.extend(["--github-query", args.github_query])
        
    if args.max_repos:
        cmd.extend(["--max-repos", str(args.max_repos)])
        
    if args.files_per_repo:
        cmd.extend(["--files-per-repo", str(args.files_per_repo)])
        
    if args.kaggle_keywords:
        cmd.extend(["--kaggle-keywords", args.kaggle_keywords])
        
    if args.docs_urls:
        cmd.extend(["--docs-urls", args.docs_urls])
    
    # Run data collection
    print(f"Running data collection: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    # Return path to collected data
    return Path(args.output_dir) / "clean_dataset.jsonl"

def run_instruction_generation(input_path, args):
    """Run the instruction generation step"""
    print("\n=== Step 2: Instruction Generation ===")
    
    # Ensure code_instruction_generator.py exists and is executable
    generator_script = Path("code_instruction_generator.py")
    if not check_script_exists(generator_script):
        print(f"Making {generator_script} executable...")
        make_executable(generator_script)
    
    # Output path for formatted examples
    output_path = Path(args.output_dir) / "deepseek_formatted_dataset.txt"
    
    # Build command
    cmd = [
        "./code_instruction_generator.py",
        "--input", str(input_path),
        "--output", str(output_path)
    ]
    
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
        
    if args.api_url:
        cmd.extend(["--api-url", args.api_url])
        
    if args.model:
        cmd.extend(["--model", args.model])
        
    if args.sample:
        cmd.extend(["--sample", str(args.sample)])
    
    # Run instruction generation
    print(f"Running instruction generation: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    
    return output_path

def run_fine_tuning(input_path, args):
    """Run the fine-tuning step"""
    print("\n=== Step 3: Fine-tuning ===")
    
    # Ensure deepseek_finetuner.py exists and is executable
    finetuner_script = Path("deepseek_finetuner.py")
    if not check_script_exists(finetuner_script):
        print(f"Making {finetuner_script} executable...")
        make_executable(finetuner_script)
    
    # Build command
    cmd = [
        "./deepseek_finetuner.py",
        "--input", str(input_path),
        "--model-name", args.model_name
    ]
    
    if args.base_model:
        cmd.extend(["--base-model", args.base_model])
        
    if args.epochs:
        cmd.extend(["--epochs", str(args.epochs)])
        
    if args.learning_rate:
        cmd.extend(["--learning-rate", str(args.learning_rate)])
        
    if args.train_split:
        cmd.extend(["--train-split", str(args.train_split)])
        
    if args.test:
        cmd.extend(["--test"])
    
    # Run fine-tuning
    print(f"Running fine-tuning: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Complete pipeline for DeepSeek-Coder fine-tuning")
    
    # General options
    parser.add_argument("--output-dir", type=str, default="deepseek_finetuning",
                        help="Output directory for all artifacts")
    parser.add_argument("--skip-collection", action="store_true",
                        help="Skip data collection (assume it's already done)")
    parser.add_argument("--skip-instruction", action="store_true",
                        help="Skip instruction generation (assume it's already done)")
    parser.add_argument("--collected-data", type=str, default=None,
                        help="Path to already collected data (when skipping collection)")
    parser.add_argument("--formatted-data", type=str, default=None,
                        help="Path to already formatted data (when skipping instruction)")
    
    # Data collection options
    parser.add_argument("--github-token", type=str, default=None,
                        help="GitHub API token for data collection")
    parser.add_argument("--github-query", type=str, default="language:python machine learning",
                        help="GitHub search query")
    parser.add_argument("--max-repos", type=int, default=10,
                        help="Maximum repositories to process")
    parser.add_argument("--files-per-repo", type=int, default=5,
                        help="Maximum files per repository")
    parser.add_argument("--kaggle-keywords", type=str, default="machine learning,data science",
                        help="Comma-separated Kaggle search keywords")
    parser.add_argument("--docs-urls", type=str, 
                        default="https://docs.python.org/3/tutorial/,https://pytorch.org/tutorials/",
                        help="Comma-separated documentation URLs")
    
    # Instruction generation options
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for the LLM (Claude, OpenAI, etc.)")
    parser.add_argument("--api-url", type=str, default="https://api.anthropic.com/v1/messages",
                        help="API URL for the LLM")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20240620",
                        help="Model to use for instruction generation")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only a sample of examples")
    
    # Fine-tuning options
    parser.add_argument("--model-name", type=str, default="deepseek-coder-finetuned",
                        help="Name for the fine-tuned model")
    parser.add_argument("--base-model", type=str, default="deepseek-coder",
                        help="Base model to use (default: deepseek-coder)")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
                        help="Learning rate for fine-tuning")
    parser.add_argument("--train-split", type=float, default=0.95,
                        help="Percentage of data to use for training")
    parser.add_argument("--test", action="store_true",
                        help="Test the model after fine-tuning")
    
    # Ollama options for instruction generation
    parser.add_argument("--instruction-model", type=str, default="llama3",
                        help="Ollama model to use for instruction generation")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Data Collection
    if not args.skip_collection:
        collected_data_path = run_data_collection(args)
    else:
        collected_data_path = args.collected_data or (Path(args.output_dir) / "clean_dataset.jsonl")
        if not Path(collected_data_path).exists():
            print(f"Error: Specified collected data file {collected_data_path} does not exist")
            sys.exit(1)
        print(f"Skipping data collection, using existing data: {collected_data_path}")
    
    # Step 2: Instruction Generation
    if not args.skip_instruction:
        formatted_data_path = run_instruction_generation(collected_data_path, args)
    else:
        formatted_data_path = args.formatted_data or (Path(args.output_dir) / "deepseek_formatted_dataset.txt")
        if not Path(formatted_data_path).exists():
            print(f"Error: Specified formatted data file {formatted_data_path} does not exist")
            sys.exit(1)
        print(f"Skipping instruction generation, using existing data: {formatted_data_path}")
    
    # Step 3: Fine-tuning
    run_fine_tuning(formatted_data_path, args)
    
    print("\n=== Pipeline Complete ===")
    print(f"Fine-tuned model name: {args.model_name}")
    print(f"To use the model: ollama run {args.model_name}")

if __name__ == "__main__":
    main()