# Example Samples from Dadiya-Harsh-wappnet/MCP-DEMO

## Example 1

### Prompt

```
### Instruction:
Implement Python code for a file named 'postgres_server.py' from the 'MCP-DEMO' repository

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
Implement Python code for a file named 'web_search_using_tavily.py' from the 'MCP-DEMO' repository

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

## Example 3

### Prompt

```
### Instruction:
Implement Python code for a file named 'type_dict_ex.py' from the 'MCP-DEMO' repository

### Response:
```

### Response

```
from typing import Annotated, TypedDict, Callable, get_type_hints


name = Annotated[str, "Name of the person"]

age = Annotated[int, "Age of the person"]   

class Person(TypedDict):
    name: name
    age: age

print(get_type_hints(Person))  # {'name': <class 'str'>, 'age': <class 'int'>} 
print(name.__metadata__)  # (str, 'Name of the person')
```

--------------------------------------------------------------------------------

## Example 4

### Prompt

```
### Instruction:
Implement Python code for a file named 'TaskGroup_usage.py' from the 'MCP-DEMO' repository

### Response:
```

### Response

```
import asyncio

async def fetch_data(id_number: int, delay: int) -> str:
    """
    Simulate a network call to fetch data
    :param:
        id_number: id of the task
        delay: time to wait before returning data
    :return:
        string that contains how much time it took to fetch data
    """

    print(f"Task {id_number}: Getting data...")
    await asyncio.sleep(delay)
    print(f"Task {id_number}: Data received!")
    return f"Task {id_number}: Data after {delay} seconds"

async de...
```

--------------------------------------------------------------------------------

## Example 5

### Prompt

```
### Instruction:
Implement Python code for a file named 'async_1.py' from the 'MCP-DEMO' repository

### Response:
```

### Response

```
import asyncio

async def main() -> None:
    """
    whenever async function is called, it returns a coroutine object
    :return:
    """

    print("Hello from mcp-demo!")

if __name__ == "__main__":
    asyncio.run(main())
```

--------------------------------------------------------------------------------

