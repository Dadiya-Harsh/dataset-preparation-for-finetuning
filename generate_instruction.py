#!/usr/bin/env python3
"""
Code Instruction Generator using Ollama

This script processes code samples and generates high-quality instruction-code 
pairs using local Ollama models instead of paid API services for fine-tuning
DeepSeek-Coder on Ollama.
"""

import os
import json
import argparse
import random
import time
import re
import subprocess
from pathlib import Path
from tqdm import tqdm
import multiprocessing
from functools import partial

class OllamaInstructionGenerator:
    def __init__(self, input_path, output_path, model="llama3.2"):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(exist_ok=True, parents=True)
        
        # Ollama settings
        self.model = model
        
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
    
    def check_ollama_available(self):
        """Check if Ollama is installed and available"""
        try:
            result = subprocess.run(["ollama", "list"], 
                                    stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def generate_instruction_with_ollama(self, code, language):
        """Generate a detailed instruction for code using Ollama"""
        try:
            # Prepare prompt
            prompt = self.instruction_prompt_template.format(
                code=code,
                language=language or "python"
            )
            
            # Call Ollama
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                instruction = result.stdout.strip()
                
                # Clean up the instruction (remove any code block formatting)
                instruction = re.sub(r'```.*?```', '', instruction, flags=re.DOTALL)
                instruction = re.sub(r'Here is your instruction:', '', instruction)
                instruction = re.sub(r'Here is the instruction:', '', instruction)
                instruction = re.sub(r'Instruction:', '', instruction)
                
                # Remove any lines that look like they're part of a conversation
                lines = instruction.split('\n')
                filtered_lines = []
                for line in lines:
                    if not line.strip().startswith(('You:', 'I:', 'Human:', 'Assistant:')):
                        filtered_lines.append(line)
                
                instruction = '\n'.join(filtered_lines).strip()
                
                return instruction
            else:
                print(f"Ollama command failed: {result.stderr}")
                return self._generate_basic_instruction(code, language)
                
        except subprocess.TimeoutExpired:
            print("Ollama instruction generation timed out")
            return self._generate_basic_instruction(code, language)
        except Exception as e:
            print(f"Error generating instruction with Ollama: {str(e)}")
            return self._generate_basic_instruction(code, language)
    
    def enhance_existing_description(self, description, code, language):
        """Enhance an existing simple description using Ollama"""
        try:
            # Prepare prompt
            prompt = self.enhance_description_prompt.format(
                description=description,
                code=code,
                language=language or "python"
            )
            
            # Call Ollama
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                enhanced = result.stdout.strip()
                
                # Clean up the instruction
                enhanced = re.sub(r'```.*?```', '', enhanced, flags=re.DOTALL)
                enhanced = re.sub(r'Here is your enhanced instruction:', '', enhanced)
                enhanced = re.sub(r'Here is the enhanced instruction:', '', enhanced)
                enhanced = re.sub(r'Enhanced instruction:', '', enhanced)
                
                # Remove any lines that look like they're part of a conversation
                lines = enhanced.split('\n')
                filtered_lines = []
                for line in lines:
                    if not line.strip().startswith(('You:', 'I:', 'Human:', 'Assistant:')):
                        filtered_lines.append(line)
                
                enhanced = '\n'.join(filtered_lines).strip()
                
                return enhanced
            else:
                print(f"Ollama command failed: {result.stderr}")
                return description
                
        except subprocess.TimeoutExpired:
            print("Ollama instruction enhancement timed out")
            return description
        except Exception as e:
            print(f"Error enhancing description with Ollama: {str(e)}")
            return description
    
    def _generate_basic_instruction(self, code, language):
        """Generate a basic instruction without using Ollama (fallback)"""
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
        ollama_available = self.check_ollama_available()
        
        if ollama_available:
            # If Ollama is available, use it to generate instructions
            if len(original_instruction) > 10 and not original_instruction.startswith("Implement"):
                # If there's a decent existing instruction, enhance it
                instruction = self.enhance_existing_description(
                    original_instruction, response_code, language)
            else:
                # Otherwise generate from scratch
                instruction = self.generate_instruction_with_ollama(response_code, language)
                
            # Sleep to avoid overwhelming the local system
            time.sleep(0.5)
        else:
            # Fallback to basic instruction generation
            print("Warning: Ollama not available. Using basic instruction generation.")
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

    def process_examples_sequential(self, examples):
        """Process examples sequentially (Ollama can't handle parallel well)"""
        results = []
        
        for example in tqdm(examples, desc="Processing examples"):
            results.append(self.process_example(example))
                
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
        # Check if Ollama is available
        if not self.check_ollama_available():
            print("Warning: Ollama is not installed or not in PATH.")
            print("Will use basic instruction generation instead.")
        else:
            print(f"Using Ollama model '{self.model}' for instruction generation")
        
        # Load examples
        examples = self.read_input_data()
        print(f"Loaded {len(examples)} examples from {self.input_path}")
        
        # Sample if requested
        if sample_size and len(examples) > sample_size:
            examples = random.sample(examples, sample_size)
            print(f"Sampled {sample_size} examples for processing")
            
        # Process examples
        processed = self.process_examples_sequential(examples)
        
        # Write output
        self.write_output(processed)
        
        return processed

def main():
    parser = argparse.ArgumentParser(description="Generate code-instruction pairs using Ollama")
    parser.add_argument("--input", type=str, required=True, 
                        help="Input JSONL file with code examples")
    parser.add_argument("--output", type=str, required=True,
                        help="Output file for processed examples")
    parser.add_argument("--model", type=str, default="llama3",
                        help="Ollama model to use for instruction generation")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only a sample of examples")
    
    args = parser.parse_args()
    
    generator = OllamaInstructionGenerator(
        input_path=args.input,
        output_path=args.output,
        model=args.model
    )
    
    generator.run(sample_size=args.sample)

if __name__ == "__main__":
    main()