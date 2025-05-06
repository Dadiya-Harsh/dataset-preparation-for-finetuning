import json
import re
import requests
from typing import List, Optional, Tuple, Dict
from firecrawl import FirecrawlApp, ScrapeOptions
from dotenv import load_dotenv
import os
import time

# Load environment variables
load_dotenv()

# Initialize FireCrawl app
FIRECRWAL_API_KEY = os.getenv("FIRECRWAL_API_KEY")
if not FIRECRWAL_API_KEY:
    raise ValueError("FIRECRWAL_API_KEY is missing in .env file")

app = FirecrawlApp(api_key=FIRECRWAL_API_KEY)

# === Utility Functions ===

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


def crawl_website(url: str, max_depth: int = 2, limit: int = 50) -> Optional[List[Dict]]:
    """
    Starts crawling a website and waits for completion.
    Returns list of crawled documents (each containing markdown).
    """
    try:
        print(f"[+] Starting crawl for URL: {url}, Max Depth: {max_depth}, Limit: {limit}")
        result = app.crawl_url(url, max_depth=max_depth, limit=limit, scrape_options=ScrapeOptions(formats=["markdown"]))
        
        if not result.success:
            print(f"[!] Crawl job failed: {result.error}")
            return None
        
        print(f"[✓] Crawl completed. Total pages scraped: {len(result.data)}")
        return result.data  # List of crawled documents

    except Exception as e:
        print(f"[!] Error during crawling: {e}")
        return None

def process_crawled_data(crawled_pages: List[Dict], model: str = "llama3.2") -> List[Tuple[str, str]]:
    """
    Process crawled pages to extract markdown, clean, and generate instruction-response pairs.
    """
    pairs = []
    for idx, page in enumerate(crawled_pages):
        try:
            # Access attributes directly from FirecrawlDocument object
            url = getattr(page, "url", "unknown")
            markdown = getattr(page, "markdown", "")
        except AttributeError:
            print(f"[!] Skipping invalid page data at index {idx}")
            continue

        if not markdown:
            print(f"[!] No markdown found for page {idx+1} ({url})")
            continue

        cleaned_md = clean_markdown_content(markdown)
        paragraphs = split_into_paragraphs(cleaned_md)

        for para in paragraphs:
            instruction = generate_instruction(para, model)
            if instruction:
                pairs.append((instruction, para))
            else:
                print(f"[!] Failed to generate instruction for paragraph from {url}")

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


# === Main Execution ===

if __name__ == "__main__":
    target_url = "https://firecrawl.dev"
    max_depth = 2  # How deep to follow internal links
    page_limit = 50  # Maximum number of pages to crawl

    # Step 1: Crawl the website
    crawled_data = crawl_website(target_url, max_depth=max_depth, limit=page_limit)
    
    if not crawled_data:
        print("[-] No data was crawled.")
    else:
        # Step 2: Generate instruction-response pairs
        print(f"[+] Processing {len(crawled_data)} crawled pages...")
        instruction_response_pairs = process_crawled_data(crawled_data)

        # Step 3: Save to file
        if instruction_response_pairs:
            save_instruction_response_pairs(instruction_response_pairs, "output.jsonl")
        else:
            print("[-] No instruction-response pairs generated.")

    print("[*] Operation completed.")