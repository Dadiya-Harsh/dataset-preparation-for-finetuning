# Example Samples from Dadiya-Harsh/mcp-server-suite

## Example 1

### Prompt

```
### Instruction:
Implement Python code for a file named 'sync_postgressql_server.py' from the 'mcp-server-suite' repository

### Response:
```

### Response

```
import os

import psycopg2
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from psycopg2.extras import RealDictCursor

# Load environment variables from .env file
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(
    name="postgres_server",
    port=8001,
    host="localhost",
    description="Access PostgresSQL databases by name and perform SQL operations"
)

# Database connection pool
db_connections = {}

@mcp.resource(name="postgres_server", description="Provides...
```

--------------------------------------------------------------------------------

## Example 2

### Prompt

```
### Instruction:
Implement Python code for a file named 'web_search_stdio.py' from the 'mcp-server-suite' repository

### Response:
```

### Response

```
from mcp.server.fastmcp import FastMCP
from tavily import TavilyClient
import os
from dotenv import load_dotenv

load_dotenv()

mcp = FastMCP(name = "Web Search", host = "localhost", port = 8000)


@mcp.tool()
async def web_search(query: str) -> dict:
    """Perform a web search using the Tavily API.

    Args:
        query: The search query string.

    Returns:
        A list of URLs from the search results.
    """
    # Simulate a web search using Tavily API
    tavily_client = TavilyClient...
```

--------------------------------------------------------------------------------

## Example 3

### Prompt

```
### Instruction:
Implement Python code for a file named 'web_serach_sse.py' from the 'mcp-server-suite' repository

### Response:
```

### Response

```
from mcp.server.fastmcp import FastMCP, Context
from mcp.types import TextContent
from tavily import TavilyClient
import os
from dotenv import load_dotenv
import json

load_dotenv()

mcp = FastMCP(
    name="web_search_server",
    host="localhost",
    port=8000,
    log_level="INFO",
)

@mcp.tool(
    name="web_search",
    description="Perform a web search using the Tavily API."
)
async def web_search(query: str, ctx: Context) -> list[TextContent]:
    """
    Perform a web search using the T...
```

--------------------------------------------------------------------------------

