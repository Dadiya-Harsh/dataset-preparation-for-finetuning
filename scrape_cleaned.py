import json
import re
import traceback
import requests
from typing import List, Optional, Tuple
from firecrawl import FirecrawlApp
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize FireCrawl app
FIRECRWAL_API_KEY = os.getenv("FIRECRWAL_API_KEY")
if not FIRECRWAL_API_KEY:
    raise ValueError("FIRECRWAL_API_KEY is missing in .env file")

app = FirecrawlApp(api_key=FIRECRWAL_API_KEY)

def generate_instruction(response_text: str, model: str = "llama3.2") -> Optional[str]:
    """
    Generates an instruction for a given response text using a local Ollama model.
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
        return response.json().get("response", "").strip()
    except Exception as e:
        print(f"[!] Instruction generation failed: {e}")
        return None


def clean_markdown_content(markdown: str) -> str:
    """
    Cleans markdown by removing HTML tags, image links, special characters, and normalizing whitespace.
    """
    if not markdown:
        return ""

    # Remove HTML tags
    markdown = re.sub(r"<[^>]+>", "", markdown)

    # Remove image links
    markdown = re.sub(r"!\[.*?\]\(.*?\)", "", markdown)

    # Replace escaped quotes and backslashes
    markdown = markdown.replace('\\"', '"').replace("\\'", "'").replace('\\\\', '\\')

    # Remove Unicode escape sequences like \\xa0
    markdown = re.sub(r"\\x[a-fA-F0-9]{2}", " ", markdown)

    # Normalize spaces and line breaks
    markdown = re.sub(r"\s+", " ", markdown).strip()

    return markdown


def split_into_paragraphs(text: str, min_length: int = 100) -> List[str]:
    """
    Splits cleaned markdown into paragraphs based on double newlines or heuristics.
    Only returns paragraphs longer than min_length.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if len(p.strip()) > min_length]


def scrape_website(url: str) -> Optional[str]:
    """
    Scrapes the website and returns cleaned markdown content.
    """
    try:
        result = app.scrape_url(url, formats=['markdown'])

        if not hasattr(result, 'success'):
            print("[!] Unexpected response from FireCrawl, 'success' field missing.")
            return None

        if result.success:
            markdown_content = getattr(result, "markdown", "")
            return clean_markdown_content(markdown_content)
        else:
            error_msg = getattr(result, "error", "Unknown error")
            print(f"[!] Scraping failed: {error_msg}")
            return None

    except Exception as e:
        print(f"[!] Error during scraping: {e}")
        traceback.print_exc()
        return None


def generate_instruction_response_pairs(content: str, model: str = "llama3.2") -> List[Tuple[str, str]]:
    """
    Generate instruction + response pair for each paragraph of content.
    Returns list of valid pairs.
    """
    pairs = []
    paragraphs = split_into_paragraphs(content)

    for para in paragraphs:
        instruction = generate_instruction(para, model)
        if instruction:
            pairs.append((instruction, para))
        else:
            print("[!] Skipping paragraph due to failed instruction generation.")

    return pairs


def save_instruction_response_pairs(data: List[Tuple[str, str]], output_file: str = "output.jsonl") -> None:
    """
    Saves the instruction-response pairs to a JSONL file.
    """
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for instruction, response in data:
                if instruction and response:
                    json.dump({
                        "instruction": instruction,
                        "response": response
                    }, f, ensure_ascii=False)
                    f.write("\n")
        print(f"[✓] Saved {len(data)} entries to {output_file}")
    except Exception as e:
        print(f"[!] Failed to save data: {e}")


# Main Execution
if __name__ == "__main__":
    url = 'https://firecrawl.dev'
    print(f"[+] Scraping URL: {url}")

    raw_content = scrape_website(url)
    if not raw_content:
        print("[-] No content scraped. Exiting...")
    else:
        print(f"[+] Successfully scraped and cleaned content. Length: {len(raw_content)} chars")

        pairs = generate_instruction_response_pairs(raw_content)
        if pairs:
            save_instruction_response_pairs(pairs, "output.jsonl")
        else:
            print("[-] No instruction-response pairs generated.")

    print("[*] Operation completed.")