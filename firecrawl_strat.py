import traceback
from typing import Optional
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os
import requests

load_dotenv()

def generate_instruction(response_text: str, model: str = "deepseek-r1:1.5b") -> Optional[str]: # change model to use resources efficiently
    """
    Generates an instruction from the given content using an Ollama model.

    Args:
        response_text (str): The scraped content (code, text, or data) from the internet.
        model (str): The local Ollama model to use (e.g., "llama3", "mistral", etc.)

    Returns:
        str or None: The generated instruction text, or None if generation fails.
    """
    prompt = f"""
You are an expert at crafting high-quality, task-specific instruction-response pairs for training large language models.

Given the following piece of content (which may be code, text, or structured data), your job is to generate a clear and specific instruction that would naturally lead a person (or model) to produce this exact content as a response.

Be specific about the task, its purpose, and expected output format — but do not repeat or paraphrase the content itself.

Content to generate instruction for:
{response_text}

✅ Your task: Write a precise, standalone instruction (prompt) that would lead someone to write the above content. Focus on intent, constraints, and clarity. Respond with only the instruction text — no explanations or formatting.
    """

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        print(f"\nHi, this is respones i got from llm \n \n {response.json().get('response')}\n and \n it's type is: {type(response)}")
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"Error during instruction generation: {e}")
        return None

        

app = FirecrawlApp(os.getenv("FIRECRWAL_API_KEY"), api_url="http://localhost:3002")

scrape_result = app.scrape_url('firecrawl.dev', formats=['markdown'])
# print(f"Scraped data is: {scrape_result}\n and it's type is: {type(scrape_result)}")
res = generate_instruction(str(scrape_result))

# print(f"\nResult to be written :{res}")

with open("example.log", mode="w", encoding="utf-8") as f:
    f.write(res)

print("Operation Completed..")