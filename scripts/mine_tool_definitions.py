#!/usr/bin/env python3
"""Mine tool definitions from external sources.

Sources:
1. LangChain tools - https://github.com/langchain-ai/langchain
2. Gorilla/Berkeley Function Calling - https://gorilla.cs.berkeley.edu/
3. ToolBench - https://github.com/OpenBMB/ToolBench
4. HuggingFace Transformers Agents tools
5. OpenAI function calling examples

Output: YAML definitions compatible with tool_registry.yaml
"""

import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

# Output directory
OUTPUT_DIR = Path("/mnt/raid0/llm/claude/orchestration/tools/mined")


def fetch_json(url: str) -> Any:
    """Fetch and parse JSON from URL."""
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def fetch_jsonl(url: str) -> list:
    """Fetch and parse JSONL (JSON Lines) from URL."""
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            content = response.read().decode()
            lines = content.strip().split('\n')
            return [json.loads(line) for line in lines if line.strip()]
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return []


def mine_gorilla_berkeley():
    """Mine tools from Berkeley Function Calling Leaderboard dataset (BFCL v4)."""
    print("\n=== Mining Gorilla/Berkeley Function Calling (BFCL v4) ===")

    # BFCL v4 test data - corrected URLs (Jan 2026)
    base_url = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data"
    files = [
        "BFCL_v4_simple_python.json",
        "BFCL_v4_multiple.json",
        "BFCL_v4_parallel.json",
    ]

    tools = {}

    for filename in files:
        url = f"{base_url}/{filename}"
        print(f"  Fetching {filename}...")
        data = fetch_jsonl(url)  # BFCL uses JSONL format
        if not data:
            continue

        # BFCL v4 format: each entry has {id, question, function}
        for item in data[:200]:  # Sample first 200 entries per file
            funcs = item.get("function", [])
            if isinstance(funcs, dict):
                funcs = [funcs]

            for func in funcs:
                name = func.get("name", "")
                if not name or name in tools:
                    continue

                # Extract parameters from the schema
                params = func.get("parameters", {})
                # BFCL uses "properties" inside parameters
                properties = params.get("properties", {})
                required = params.get("required", [])

                # Convert to simpler format
                simple_params = {}
                for pname, pspec in properties.items():
                    simple_params[pname] = {
                        "type": pspec.get("type", "string"),
                        "description": pspec.get("description", ""),
                    }
                    if pname in required:
                        simple_params[pname]["required"] = True
                    if "default" in pspec:
                        simple_params[pname]["default"] = pspec["default"]
                    if "enum" in pspec:
                        simple_params[pname]["enum"] = pspec["enum"]

                # Categorize by name heuristics
                category = "general"
                name_lower = name.lower()
                if any(w in name_lower for w in ["search", "fetch", "get_", "http", "api"]):
                    category = "web"
                elif any(w in name_lower for w in ["calculate", "math", "compute", "area", "volume"]):
                    category = "math"
                elif any(w in name_lower for w in ["file", "read", "write", "directory"]):
                    category = "system"
                elif any(w in name_lower for w in ["sql", "database", "query", "stock", "price"]):
                    category = "data"
                elif any(w in name_lower for w in ["email", "calendar", "schedule", "message"]):
                    category = "interaction"
                elif any(w in name_lower for w in ["code", "execute", "run", "python"]):
                    category = "code"

                tools[name] = {
                    "name": name,
                    "description": func.get("description", ""),
                    "parameters": simple_params,
                    "source": "bfcl_v4",
                    "category": category,
                }

    print(f"  Found {len(tools)} unique tools")
    return tools


def mine_toolbench():
    """Mine tools from ToolBench dataset."""
    print("\n=== Mining ToolBench ===")

    # ToolBench API definitions
    url = "https://raw.githubusercontent.com/OpenBMB/ToolBench/main/data/toolenv/tools/Travel/booking_com/api.json"

    tools = {}
    data = fetch_json(url)

    if data and isinstance(data, dict):
        for name, spec in data.items():
            if isinstance(spec, dict):
                tools[name] = {
                    "name": name,
                    "description": spec.get("description", ""),
                    "parameters": spec.get("parameters", {}),
                    "source": "toolbench",
                }

    print(f"  Found {len(tools)} tools")
    return tools


def mine_langchain_tools():
    """Extract tool definitions from LangChain documentation/code."""
    print("\n=== Mining LangChain Tools (hardcoded list) ===")

    # LangChain built-in tools (from documentation)
    tools = {
        "duckduckgo_search": {
            "name": "duckduckgo_search",
            "description": "Search the web using DuckDuckGo",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            },
            "source": "langchain",
            "category": "web",
        },
        "wikipedia": {
            "name": "wikipedia",
            "description": "Search and retrieve Wikipedia articles",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            },
            "source": "langchain",
            "category": "web",
        },
        "arxiv": {
            "name": "arxiv",
            "description": "Search arXiv for academic papers",
            "parameters": {
                "query": {"type": "string", "description": "Search query"},
                "max_results": {"type": "integer", "default": 5}
            },
            "source": "langchain",
            "category": "web",
        },
        "pubmed": {
            "name": "pubmed",
            "description": "Search PubMed for medical/biomedical literature",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            },
            "source": "langchain",
            "category": "web",
        },
        "wolfram_alpha": {
            "name": "wolfram_alpha",
            "description": "Query Wolfram Alpha for computational knowledge",
            "parameters": {
                "query": {"type": "string", "description": "Query to compute"}
            },
            "source": "langchain",
            "category": "math",
        },
        "python_repl": {
            "name": "python_repl",
            "description": "Execute Python code in a REPL",
            "parameters": {
                "code": {"type": "string", "description": "Python code to execute"}
            },
            "source": "langchain",
            "category": "code",
        },
        "shell": {
            "name": "shell",
            "description": "Execute shell commands",
            "parameters": {
                "command": {"type": "string", "description": "Shell command"}
            },
            "source": "langchain",
            "category": "code",
        },
        "requests_get": {
            "name": "requests_get",
            "description": "Make HTTP GET request",
            "parameters": {
                "url": {"type": "string", "description": "URL to fetch"}
            },
            "source": "langchain",
            "category": "web",
        },
        "requests_post": {
            "name": "requests_post",
            "description": "Make HTTP POST request",
            "parameters": {
                "url": {"type": "string", "description": "URL to post to"},
                "data": {"type": "object", "description": "JSON data to send"}
            },
            "source": "langchain",
            "category": "web",
        },
        "file_search": {
            "name": "file_search",
            "description": "Search for files matching a pattern",
            "parameters": {
                "pattern": {"type": "string", "description": "Glob pattern"},
                "directory": {"type": "string", "default": "."}
            },
            "source": "langchain",
            "category": "system",
        },
        "read_file": {
            "name": "read_file",
            "description": "Read contents of a file",
            "parameters": {
                "path": {"type": "string", "description": "File path"}
            },
            "source": "langchain",
            "category": "system",
        },
        "write_file": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "path": {"type": "string", "description": "File path"},
                "content": {"type": "string", "description": "Content to write"}
            },
            "source": "langchain",
            "category": "system",
        },
        "youtube_search": {
            "name": "youtube_search",
            "description": "Search YouTube for videos",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            },
            "source": "langchain",
            "category": "web",
        },
        "google_serper": {
            "name": "google_serper",
            "description": "Search Google using Serper API",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            },
            "source": "langchain",
            "category": "web",
        },
        "bing_search": {
            "name": "bing_search",
            "description": "Search using Bing API",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            },
            "source": "langchain",
            "category": "web",
        },
        "human": {
            "name": "human",
            "description": "Ask a human for input or clarification",
            "parameters": {
                "question": {"type": "string", "description": "Question to ask"}
            },
            "source": "langchain",
            "category": "interaction",
        },
    }

    print(f"  Found {len(tools)} tools")
    return tools


def mine_openai_cookbook():
    """Extract function definitions from OpenAI Cookbook examples."""
    print("\n=== Mining OpenAI Cookbook Functions ===")

    # Common function calling examples
    tools = {
        "get_current_weather": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "location": {"type": "string", "description": "City and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            },
            "source": "openai_cookbook",
            "category": "web",
        },
        "search_bing": {
            "name": "search_bing",
            "description": "Search the web using Bing",
            "parameters": {
                "query": {"type": "string", "description": "Search query"}
            },
            "source": "openai_cookbook",
            "category": "web",
        },
        "get_stock_price": {
            "name": "get_stock_price",
            "description": "Get current stock price for a ticker symbol",
            "parameters": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"}
            },
            "source": "openai_cookbook",
            "category": "data",
        },
        "send_email": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "to": {"type": "string", "description": "Email address"},
                "subject": {"type": "string"},
                "body": {"type": "string"}
            },
            "source": "openai_cookbook",
            "category": "interaction",
        },
        "create_calendar_event": {
            "name": "create_calendar_event",
            "description": "Create a calendar event",
            "parameters": {
                "title": {"type": "string"},
                "start_time": {"type": "string", "description": "ISO datetime"},
                "end_time": {"type": "string"},
                "description": {"type": "string"}
            },
            "source": "openai_cookbook",
            "category": "interaction",
        },
        "execute_sql": {
            "name": "execute_sql",
            "description": "Execute a SQL query against the database",
            "parameters": {
                "query": {"type": "string", "description": "SQL query to execute"}
            },
            "source": "openai_cookbook",
            "category": "data",
        },
    }

    print(f"  Found {len(tools)} tools")
    return tools


def mine_hf_agents():
    """Extract tool definitions from HuggingFace Agents."""
    print("\n=== Mining HuggingFace Agents Tools ===")

    tools = {
        "image_captioner": {
            "name": "image_captioner",
            "description": "Generate a caption for an image",
            "parameters": {
                "image": {"type": "string", "description": "Path or URL to image"}
            },
            "source": "hf_agents",
            "category": "vision",
        },
        "image_qa": {
            "name": "image_qa",
            "description": "Answer questions about an image",
            "parameters": {
                "image": {"type": "string"},
                "question": {"type": "string"}
            },
            "source": "hf_agents",
            "category": "vision",
        },
        "text_to_image": {
            "name": "text_to_image",
            "description": "Generate an image from a text description",
            "parameters": {
                "prompt": {"type": "string", "description": "Image description"}
            },
            "source": "hf_agents",
            "category": "vision",
        },
        "document_qa": {
            "name": "document_qa",
            "description": "Answer questions about a document",
            "parameters": {
                "document": {"type": "string", "description": "Document text or path"},
                "question": {"type": "string"}
            },
            "source": "hf_agents",
            "category": "data",
        },
        "text_summarizer": {
            "name": "text_summarizer",
            "description": "Summarize a piece of text",
            "parameters": {
                "text": {"type": "string"},
                "max_length": {"type": "integer", "default": 150}
            },
            "source": "hf_agents",
            "category": "llm",
        },
        "text_classifier": {
            "name": "text_classifier",
            "description": "Classify text into categories",
            "parameters": {
                "text": {"type": "string"},
                "labels": {"type": "array", "description": "Possible labels"}
            },
            "source": "hf_agents",
            "category": "llm",
        },
        "translator": {
            "name": "translator",
            "description": "Translate text between languages",
            "parameters": {
                "text": {"type": "string"},
                "src_lang": {"type": "string"},
                "tgt_lang": {"type": "string"}
            },
            "source": "hf_agents",
            "category": "llm",
        },
        "speech_to_text": {
            "name": "speech_to_text",
            "description": "Transcribe audio to text",
            "parameters": {
                "audio": {"type": "string", "description": "Path to audio file"}
            },
            "source": "hf_agents",
            "category": "audio",
        },
        "text_to_speech": {
            "name": "text_to_speech",
            "description": "Convert text to speech audio",
            "parameters": {
                "text": {"type": "string"}
            },
            "source": "hf_agents",
            "category": "audio",
        },
    }

    print(f"  Found {len(tools)} tools")
    return tools


def save_tools(tools: dict[str, dict], filename: str):
    """Save tools to YAML file."""
    import yaml

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w") as f:
        yaml.dump({"tools": tools}, f, default_flow_style=False, sort_keys=False)

    print(f"  Saved to {output_path}")


def main():
    print("=" * 60)
    print("MINING TOOL DEFINITIONS FROM EXTERNAL SOURCES")
    print("=" * 60)

    all_tools = {}

    # Mine from each source
    sources = [
        ("gorilla_berkeley.yaml", mine_gorilla_berkeley),
        ("langchain.yaml", mine_langchain_tools),
        ("openai_cookbook.yaml", mine_openai_cookbook),
        ("hf_agents.yaml", mine_hf_agents),
        # ("toolbench.yaml", mine_toolbench),  # Often slow/unavailable
    ]

    for filename, miner in sources:
        try:
            tools = miner()
            if tools:
                save_tools(tools, filename)
                all_tools.update(tools)
        except Exception as e:
            print(f"  Error: {e}")

    # Save combined file
    if all_tools:
        save_tools(all_tools, "all_mined_tools.yaml")

    print("\n" + "=" * 60)
    print(f"TOTAL: {len(all_tools)} unique tool definitions mined")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
