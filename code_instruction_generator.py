#!/usr/bin/env python3
"""
Code Instruction Generator for DeepSeek-Coder Fine-tuning

This script processes code samples (collected with data_collector.py) and 
generates high-quality instruction-code pairs using an LLM API for fine-tuning
DeepSeek-Coder on Ollama.

Note: This version uses external APIs like Claude/OpenAI. If you need a free option,
use ollama_instruction_generator.py instead, which uses locally running Ollama models.
"""

import os
import json
import argparse
import random
import time
import re
from pathlib import Path
from tqdm import tqdm
import requests
import multiprocessing
from functools import partial

class CodeInstructionGenerator:
    def __init__(self, input_path, output_path, api_key=None, api_url=None, model=None):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # API settings for the LLM
        self.api_key = api_key
        self.api_url = api_url or "https://api.anthropic.com/v1/messages"
        self.model = model or "claude-3-5-sonnet-20240620"
        
        # Template for DeepSeek-Coder format
        self.deepseek_template = """<|im_start|>user
{instruction}
<|im_end|>
<|im_start|>assistant
{code}
<|im_end|>"""
        
        # Instruction generation prompt templates
        self.instruction_prompt_template = """You are an expert programmer tasked with creating detailed and educational programming task instructions.

I'll provide you with a code sample, and I need you to create a detailed instruction that would guide someone to write this exact code.

The instruction should:
1. Be clear and specific about what to implement
2. Include the purpose and functionality of the code
3. Specify key components, classes, or functions to include
4. Mention any important algorithms or techniques to use
5. Provide context about when this code would be useful
6. Be written as if you're a programming instructor giving an assignment

Don't just describe what the code does - write an instruction that would lead someone to create this code from scratch.

Here's the code:
```{language}
{code}
```

Generate only the instruction, without any additional commentary. The instruction should be substantial (at least 3-4 sentences) and detailed enough to guide implementation."""

        # Enhance existing description prompt
        self.enhance_description_prompt = """You are an expert programming instructor. I'll provide you with a simple description of code and the actual code. Your task is to transform the simple description into a detailed, educational instruction that would guide a programmer to write this exact code.

Original description: {description}

Code:
```{language}
{code}
```

Generate a much more detailed instruction (at least 3-4 sentences) that:
1. Explains the purpose and context of the code
2. Specifies what to implement in detail
3. Mentions important components, classes, and functions
4. Includes any relevant techniques or patterns to use
5. Provides guidance on implementation approach

Write only the enhanced instruction, no additional commentary."""

    def read_input_data(self):
        """Read the input JSONL file with code examples"""
        examples = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                examples.append(json.loads(line))
        return examples
    
    def generate_instruction_with_llm(self, code, language):
        """Generate a detailed instruction for code using an LLM API"""
        if not self.api_key:
            # Fallback if no API key is provided
            return self._generate_basic_instruction(code, language)
            
        try:
            prompt = self.instruction_prompt_template.format(
                code=code,
                language=language or "python"
            )
            
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                instruction = result["content"][0]["text"]
                return instruction.strip()
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                return self._generate_basic_instruction(code, language)
                
        except Exception as e:
            print(f"Error generating instruction: {str(e)}")
            return self._generate_basic_instruction(code, language)
    
    def enhance_existing_description(self, description, code, language):
        """Enhance an existing simple description using an LLM API"""
        if not self.api_key:
            return description
            
        try:
            prompt = self.enhance_description_prompt.format(
                description=description,
                code=code,
                language=language or "python"
            )
            
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json"
            }
            
            data = {
                "model": self.model,
                "max_tokens": 1024,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                enhanced = result["content"][0]["text"]
                return enhanced.strip()
            else:
                print(f"API request failed with status {response.status_code}: {response.text}")
                return description
                
        except Exception as e:
            print(f"Error enhancing description: {str(e)}")
            return description
    
    def _generate_basic_instruction(self, code, language):
        """Generate a basic instruction without using an API (fallback)"""
        # Extract function/class names
        functions = []
        classes = []
        if language == "Python":
            functions = re.findall(r'def\s+([a-zA-Z0-9_]+)\s*\(', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[:\(]', code)
        elif language in ["JavaScript", "TypeScript"]:
            functions = re.findall(r'function\s+([a-zA-Z0-9_]+)\s*\(', code)
            classes = re.findall(r'class\s+([a-zA-Z0-9_]+)\s*[{\(]', code)
        
        # Extract comments to understand purpose
        purpose = ""
        if language == "Python":
            docstrings = re.findall(r'"""(.+?)"""', code, re.DOTALL)
            if docstrings:
                purpose = f" The code should {docstrings[0].strip().split('.')[0].lower()}."
        
        # Build instruction
        instruction = f"Implement a {language} program"
        
        if classes:
            class_list = ", ".join(classes[:3])
            instruction += f" with {class_list} class{'es' if len(classes) > 1 else ''}"
            
        if functions:
            func_list = ", ".join(functions[:3])
            instruction += f" that includes {func_list} function{'s' if len(functions) > 1 else ''}"
            
        if purpose:
            instruction += purpose
            
        instruction += f" Use best practices and clear documentation in your {language} implementation."
        
        return instruction

    def process_example(self, example):
        """Process a single example to generate instruction"""
        response_code = example["response"]
        metadata = example.get("metadata", {})
        language = metadata.get("language", "Python")
        
        # Extract the instruction part from the prompt
        original_prompt = example["prompt"]
        original_instruction = original_prompt.split("### Instruction:\n")[-1].split("\n\n### Response:")[0]
        
        # Generate new instruction
        if self.api_key:
            # If we have an API key, generate a new detailed instruction
            if len(original_instruction) > 10 and not original_instruction.startswith("Implement"):
                # If there's a decent existing instruction, enhance it
                instruction = self.enhance_existing_description(
                    original_instruction, response_code, language)
            else:
                # Otherwise generate from scratch
                instruction = self.generate_instruction_with_llm(response_code, language)
                
            # Sleep to avoid rate limiting
            time.sleep(0.5)
        else:
            # Fallback to basic instruction generation
            instruction = self._generate_basic_instruction(response_code, language)
        
        # Format in DeepSeek-Coder format
        formatted_example = self.deepseek_template.format(
            instruction=instruction,
            code=response_code
        )
        
        return {
            "formatted": formatted_example,
            "instruction": instruction,
            "code": response_code,
            "metadata": metadata
        }

    def process_examples_parallel(self, examples, max_workers=None):
        """Process examples in parallel"""
        max_workers = max_workers or min(32, os.cpu_count() + 4)
        results = []
        
        if self.api_key:
            # If using API, process sequentially to avoid rate limits
            for example in tqdm(examples, desc="Processing examples"):
                results.append(self.process_example(example))
        else:
            # Process in parallel if not using API
            with multiprocessing.Pool(max_workers) as pool:
                results = list(tqdm(
                    pool.imap(self.process_example, examples),
                    total=len(examples),
                    desc="Processing examples"
                ))
                
        return results

    def write_output(self, processed_examples):
        """Write processed examples to output files"""
        # Write the full dataset with DeepSeek-Coder format
        with open(self.output_path, 'w', encoding='utf-8') as f:
            for example in processed_examples:
                f.write(example["formatted"] + "\n\n")
        
        # Also write a JSONL file with more detail
        jsonl_path = self.output_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for example in processed_examples:
                f.write(json.dumps({
                    "instruction": example["instruction"],
                    "code": example["code"],
                    "metadata": example["metadata"]
                }) + '\n')
                
        print(f"Written {len(processed_examples)} examples to {self.output_path}")
        print(f"JSONL format also available at {jsonl_path}")
    
    def run(self, sample_size=None):
        """Run the entire processing pipeline"""
        # Load examples
        examples = self.read_input_data()
        print(f"Loaded {len(examples)} examples from {self.input_path}")
        
        # Sample if requested
        if sample_size and len(examples) > sample_size:
            examples = random.sample(examples, sample_size)
            print(f"Sampled {sample_size} examples for processing")
            
        # Process examples
        processed = self.process_examples_parallel(examples)
        
        # Write output
        self.write_output(processed)
        
        return processed

def main():
    parser = argparse.ArgumentParser(description="Generate code-instruction pairs for fine-tuning")
    parser.add_argument("--input", type=str, required=True, 
                        help="Input JSONL file with code examples")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file for processed examples")
    parser.add_argument("--api-key", type=str, default=None,
                        help="API key for the LLM (Claude, OpenAI, etc.)")
    parser.add_argument("--api-url", type=str, default="https://api.anthropic.com/v1/messages",
                        help="API URL for the LLM")
    parser.add_argument("--model", type=str, default="claude-3-5-sonnet-20240620",
                        help="Model to use for instruction generation")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only a sample of examples")
    
    args = parser.parse_args()
    
    generator = CodeInstructionGenerator(
        input_path=args.input,
        output_path=args.output,
        api_key=args.api_key,
        api_url=args.api_url,
        model=args.model
    )
    
    generator.run(sample_size=args.sample)

if __name__ == "__main__":
    main()
