import json
import re
import time
import os
import random
import asyncio
from typing import List, Dict, Tuple, Optional, Set
from urllib.parse import urljoin, urlparse
from dotenv import load_dotenv
import logging
from pathlib import Path
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# === Browser Configuration ===
HEADLESS = os.getenv("HEADLESS", "true").lower() == "true"
BROWSER_TYPE = os.getenv("BROWSER_TYPE", "chromium")  # 'chromium', 'firefox', or 'webkit'

# === Proxy Configuration ===
USE_FREE_PROXIES = os.getenv("USE_FREE_PROXIES", "false").lower() == "true"
FREE_PROXY_LIST_URL = "https://free-proxy-list.net/"

# === Model Configuration ===
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2")

class PlaywrightScraper:
    def __init__(self, headless=HEADLESS, browser_type=BROWSER_TYPE, use_free_proxies=USE_FREE_PROXIES):
        """Initialize the Playwright scraper with browser settings."""
        self.headless = headless
        self.browser_type = browser_type
        self.use_free_proxies = use_free_proxies
        self.visited_urls: Set[str] = set()
        self.rate_limit_delay = 2  # seconds between requests
        self.proxies = []
        self.browser = None
        self.context = None
        
    async def _get_free_proxies(self) -> List[Dict[str, str]]:
        """Fetch free proxies from free-proxy-list.net using Playwright."""
        proxies = []
        logger.info("Fetching free proxies...")
        
        try:
            # Create a temporary browser context for fetching proxies
            async with async_playwright() as p:
                browser = await getattr(p, self.browser_type).launch(headless=True)
                page = await browser.new_page()
                
                await page.goto(FREE_PROXY_LIST_URL)
                
                # Wait for the proxy table to load
                await page.wait_for_selector('#proxylisttable')
                
                # Extract proxy data using JavaScript evaluation
                proxy_data = await page.evaluate('''() => {
                    const rows = Array.from(document.querySelectorAll('#proxylisttable tbody tr'));
                    return rows.map(row => {
                        const cells = Array.from(row.querySelectorAll('td'));
                        return {
                            ip: cells[0].textContent,
                            port: cells[1].textContent,
                            https: cells[6].textContent
                        };
                    }).filter(proxy => proxy.https.toLowerCase() === 'yes');
                }''')
                
                for proxy in proxy_data:
                    proxies.append({
                        'server': f'http://{proxy["ip"]}:{proxy["port"]}'
                    })
                
                await browser.close()
            
            logger.info(f"Found {len(proxies)} free proxies")
            return proxies
            
        except Exception as e:
            logger.error(f"Error fetching free proxies: {str(e)}")
            return []
    
    async def setup(self):
        """Set up the browser and context."""
        playwright = await async_playwright().start()
        
        # Get browser instance based on configuration
        browser_launcher = getattr(playwright, self.browser_type)
        
        # Launch browser
        self.browser: Browser = await browser_launcher.launch(headless=self.headless)
        
        # Set up proxy if enabled
        context_options = {
            'viewport': {'width': 1280, 'height': 800},
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36',
            'java_script_enabled': True,
        }
        
        if self.use_free_proxies:
            self.proxies = await self._get_free_proxies()
            if self.proxies:
                # Use the first proxy from the list
                context_options['proxy'] = self.proxies[0]
        
        # Create new browser context with options
        self.context: BrowserContext = await self.browser.new_context(**context_options)
        
        # Configure the context
        await self.context.add_init_script("""
            // Override the navigator.webdriver property to avoid detection
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
            
            // Override permissions
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                Promise.resolve({ state: Notification.permission }) :
                originalQuery(parameters)
            );
        """)
    
    async def close(self):
        """Close the browser and context."""
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
    
    async def rotate_proxy(self):
        """Rotate to a different proxy if available."""
        if not self.use_free_proxies or not self.proxies or len(self.proxies) <= 1:
            return False
            
        # Remove current proxy from the list
        current_proxy = self.proxies.pop(0)
        
        # If we have other proxies, use the next one
        if self.proxies:
            logger.info(f"Rotating proxy. {len(self.proxies)} proxies left")
            
            # Close old context
            if self.context:
                await self.context.close()
                
            # Create new context with next proxy
            context_options = {
                'viewport': {'width': 1280, 'height': 800},
                'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36',
                'java_script_enabled': True,
                'proxy': self.proxies[0]
            }
            
            self.context = await self.browser.new_context(**context_options)
            
            # Reconfigure the context
            await self.context.add_init_script("""
                // Override the navigator.webdriver property to avoid detection
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => false,
                });
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
                );
            """)
            
            return True
        
        # If we ran out of proxies, continue without proxy
        logger.warning("No more proxies available. Continuing without proxy")
        
        # Create new context without proxy
        if self.context:
            await self.context.close()
            
        context_options = {
            'viewport': {'width': 1280, 'height': 800},
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.67 Safari/537.36',
            'java_script_enabled': True
        }
        
        self.context = await self.browser.new_context(**context_options)
        
        # Reconfigure the context
        await self.context.add_init_script("""
            // Override the navigator.webdriver property to avoid detection
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false,
            });
        """)
        
        return False
        
    async def fetch_page(self, url: str) -> Optional[str]:
        """Fetch HTML content from a URL using Playwright."""
        if url in self.visited_urls:
            logger.info(f"Already visited {url}, skipping")
            return None
            
        self.visited_urls.add(url)
        
        # Respect rate limiting
        await asyncio.sleep(self.rate_limit_delay + random.uniform(0.5, 2.0))
        
        for attempt in range(3):  # Try 3 times
            try:
                logger.info(f"Fetching {url} (attempt {attempt+1})")
                
                # Create a new page
                page: Page = await self.context.new_page()
                
                # Add random wait and mouse movements to appear more human-like
                await page.set_default_timeout(30000)  # 30 seconds timeout
                
                # Set various headers to avoid detection
                await page.set_extra_http_headers({
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'DNT': '1',  # Do Not Track
                    'Connection': 'keep-alive',
                    'Upgrade-Insecure-Requests': '1',
                })
                
                # Navigate to the URL with wait options
                response = await page.goto(url, wait_until='networkidle', timeout=45000)
                
                if response and response.status == 200:
                    # Wait a bit for JS to execute
                    await page.wait_for_timeout(1000 + random.uniform(500, 1500))
                    
                    # Random mouse movement to simulate human behavior
                    await page.mouse.move(
                        random.randint(100, 500), 
                        random.randint(100, 500)
                    )
                    await page.wait_for_timeout(random.uniform(200, 800))
                    
                    # Scroll down to load lazy content
                    await page.evaluate("""
                        window.scrollTo({
                            top: document.body.scrollHeight / 2,
                            behavior: 'smooth'
                        });
                    """)
                    await page.wait_for_timeout(1000)
                    
                    # Get the HTML content
                    content = await page.content()
                    
                    # Close the page
                    await page.close()
                    
                    logger.info(f"Successfully fetched {url}")
                    return content
                    
                elif response and (response.status == 403 or response.status == 429):
                    logger.warning(f"Blocked (status {response.status}), retrying with different approach")
                    await page.close()
                    
                    # If we're getting blocked, try rotating proxy
                    if self.use_free_proxies:
                        rotated = await self.rotate_proxy()
                        if rotated:
                            continue
                    
                    # Add longer delay before next attempt
                    await asyncio.sleep(5 * (attempt + 1))
                    
                else:
                    status = response.status if response else "Unknown"
                    logger.warning(f"Failed with status {status}")
                    await page.close()
                    await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Error fetching {url}: {str(e)}")
                await asyncio.sleep(5)
                
                # If we encounter an error, try rotating proxy
                if self.use_free_proxies:
                    await self.rotate_proxy()
        
        logger.error(f"Failed to fetch {url} after multiple attempts")
        return None
        
    async def extract_links(self, html: str, base_url: str) -> List[str]:
        """Extract links from HTML content."""
        if not html:
            return []
            
        # Use a temporary page to extract links with JavaScript support
        page = await self.context.new_page()
        await page.set_content(html)
        
        links = await page.evaluate(f"""() => {{
            const baseUrl = "{base_url}";
            const domain = new URL(baseUrl).hostname;
            const links = [];
            
            document.querySelectorAll('a[href]').forEach(a => {{
                const href = a.getAttribute('href');
                
                // Skip anchor links, javascript, etc.
                if (href.startsWith('#') || href.startsWith('javascript:') || href.startsWith('mailto:')) {{
                    return;
                }}
                
                // Convert relative URLs to absolute
                let fullUrl;
                try {{
                    fullUrl = new URL(href, baseUrl).href;
                }} catch (e) {{
                    return;
                }}
                
                // Only include links from the same domain
                if (new URL(fullUrl).hostname === domain) {{
                    links.push(fullUrl);
                }}
            }});
            
            return [...new Set(links)];  // Return unique links
        }}""")
        
        await page.close()
        return links
        
    async def extract_text_content(self, html: str) -> str:
        """Extract and clean text content from HTML using Playwright for better JS support."""
        if not html:
            return ""
            
        # Create a temporary page to process the HTML
        page = await self.context.new_page()
        await page.set_content(html)
        
        # Extract text content with JavaScript
        text_content = await page.evaluate("""() => {
            // Remove unwanted elements
            ['script', 'style', 'noscript', 'iframe', 'svg'].forEach(tag => {
                document.querySelectorAll(tag).forEach(el => el.remove());
            });
            
            // Try to identify and remove navigation, headers, footers
            ['nav', 'header', 'footer', '.nav', '.header', '.footer', '.menu', '.sidebar', 
             '.navigation', '#nav', '#header', '#footer', '#menu', '#sidebar'].forEach(selector => {
                document.querySelectorAll(selector).forEach(el => el.remove());
            });
            
            // Extract text from main content areas
            const mainContent = document.querySelector('main, #main, .main, article, .content, #content');
            
            if (mainContent) {
                return mainContent.innerText;
            }
            
            // Fallback to body content
            return document.body.innerText;
        }""")
        
        await page.close()
        
        # Clean up text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
        
    async def crawl_website(self, start_url: str, max_depth: int = 2, max_pages: int = 50) -> List[Dict]:
        """
        Crawl a website starting from the given URL up to max_depth and max_pages.
        Returns a list of dictionaries with url and content.
        """
        await self.setup()
        
        to_visit = [(start_url, 0)]  # (url, depth)
        visited = set()
        results = []
        
        logger.info(f"Starting crawl at {start_url}, max depth: {max_depth}, max pages: {max_pages}")
        
        try:
            while to_visit and len(results) < max_pages:
                url, depth = to_visit.pop(0)
                
                if url in visited:
                    continue
                    
                visited.add(url)
                
                html = await self.fetch_page(url)
                if not html:
                    continue
                    
                # Extract content
                content = await self.extract_text_content(html)
                
                if content:
                    results.append({
                        "url": url,
                        "content": content
                    })
                    logger.info(f"Collected content from {url} ({len(results)}/{max_pages})")
                
                # If we haven't reached max depth, extract and queue links
                if depth < max_depth:
                    links = await self.extract_links(html, url)
                    
                    # Add new links to the queue
                    for link in links:
                        if link not in visited:
                            to_visit.append((link, depth + 1))
        
        except Exception as e:
            logger.error(f"Error during crawling: {str(e)}")
        
        finally:
            # Always close browser resources
            await self.close()
            
        logger.info(f"Crawl completed. Collected {len(results)} pages")
        return results


def clean_content(content: str) -> str:
    """Clean the extracted content."""
    # Remove excessive whitespace
    content = re.sub(r'\s+', ' ', content)
    
    # Remove special characters
    content = re.sub(r'[^\w\s.,?!:;()\[\]{}\-"\'/$%&]', '', content)
    
    return content.strip()


def split_into_paragraphs(text: str, min_length: int = 100) -> List[str]:
    """
    Split content into paragraphs and filter by minimum length.
    """
    # Split by multiple newlines
    paragraphs = re.split(r'\n\s*\n', text)
    
    # Further split long paragraphs by single newlines
    result = []
    for p in paragraphs:
        if len(p) > 500:  # If paragraph is very long, try to split it further
            sub_paras = p.split('\n')
            result.extend(sub_paras)
        else:
            result.append(p)
    
    # Filter by minimum length and clean
    return [p.strip() for p in result if len(p.strip()) > min_length]


async def generate_instruction(response_text: str, model: str = DEFAULT_MODEL) -> Optional[str]:
    """
    Generate an instruction for a given response text using Ollama.
    """
    import aiohttp
    
    prompt = f"""
You are an expert at crafting high-quality, task-specific instruction-response pairs for training language models.

Given the following piece of content, generate a clear and specific instruction that would lead someone to produce this exact content as a response.

Be specific about the task, its purpose, and expected output format — but do not repeat the content itself.

Content to generate instruction for:
{response_text}

Your task: Write a precise, standalone instruction that would lead someone to write the above content. Focus on intent, constraints, and clarity.
    """

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_API_URL}/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=30
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("response", "").strip()
                else:
                    logger.error(f"Instruction generation failed with status {response.status}")
                    return None
                    
    except Exception as e:
        logger.error(f"Instruction generation failed: {str(e)}")
        return None


async def process_crawled_data(crawled_pages: List[Dict], model: str = DEFAULT_MODEL) -> List[Tuple[str, str]]:
    """
    Process crawled pages to extract content and generate instruction-response pairs.
    """
    pairs = []
    for idx, page in enumerate(crawled_pages):
        url = page.get("url", "unknown")
        content = page.get("content", "")
        
        if not content:
            logger.warning(f"No content found for page {idx+1} ({url})")
            continue
        
        cleaned_content = clean_content(content)
        paragraphs = split_into_paragraphs(cleaned_content)
        
        for para in paragraphs:
            instruction = await generate_instruction(para, model)
            if instruction:
                pairs.append((instruction, para))
                logger.info(f"Generated instruction-response pair from {url}")
            else:
                logger.warning(f"Failed to generate instruction for paragraph from {url}")
                
    return pairs


async def save_instruction_response_pairs(data: List[Tuple[str, str]], output_file: str = "output.jsonl") -> None:
    """
    Save instruction-response pairs to a JSONL file.
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
        logger.info(f"Saved {len(data)} entries to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save data: {str(e)}")


# === Main Execution ===

async def main():
    target_url = input("Enter the website URL to crawl: ")
    max_depth = int(input("Maximum crawl depth (default 2): ") or "2")
    page_limit = int(input("Maximum pages to crawl (default 50): ") or "50")
    output_file = input("Output file name (default output.jsonl): ") or "output.jsonl"
    use_headless = input("Run in headless mode? (y/n, default: y): ").lower() != 'n'
    use_proxies = input("Use free proxies? (y/n, default: n): ").lower() == 'y'
    
    # Update environment variables based on user input
    os.environ["HEADLESS"] = "true" if use_headless else "false"
    os.environ["USE_FREE_PROXIES"] = "true" if use_proxies else "false"
    
    # Step 1: Initialize scraper
    scraper = PlaywrightScraper(
        headless=use_headless,
        use_free_proxies=use_proxies
    )
    
    # Step 2: Crawl the website
    crawled_data = await scraper.crawl_website(target_url, max_depth=max_depth, max_pages=page_limit)
    
    if not crawled_data:
        logger.error("No data was crawled.")
    else:
        # Step 3: Generate instruction-response pairs
        logger.info(f"Processing {len(crawled_data)} crawled pages...")
        instruction_response_pairs = await process_crawled_data(crawled_data)
        
        # Step 4: Save to file
        if instruction_response_pairs:
            await save_instruction_response_pairs(instruction_response_pairs, output_file)
        else:
            logger.error("No instruction-response pairs generated.")
            
    logger.info("Operation completed.")


if __name__ == "__main__":
    asyncio.run(main())