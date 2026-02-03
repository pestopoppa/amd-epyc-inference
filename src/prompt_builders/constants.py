"""Prompt constants: tool descriptions, rules, ReAct format, vision whitelists."""

from __future__ import annotations

from typing import Any


# Default tool descriptions for Root LM
DEFAULT_ROOT_LM_TOOLS = """### Context & Files
- `context`: str - The full input context (large, do not send to LLM)
- `artifacts`: dict - Store intermediate results
- `peek(n, file_path=None)`: Return first n characters of context or file
- `grep(pattern, file_path=None)`: Search context or file with regex
- `list_dir(path)`: List directory contents, returns JSON with files/dirs
- `file_info(path)`: Get file metadata (size, type, modified date)

### Document Processing (all return JSON strings - use json.loads())
- `ocr_document(path)`: Extract text from PDF, returns JSON with full_text, pages, figures
- `analyze_figure(image_path, prompt)`: Analyze image with vision model
- `extract_figure(pdf_path, page, bbox)`: Crop figure from PDF, returns image path

### Web & Shell
- `web_fetch(url)`: Fetch web content
- `run_shell(cmd)`: Run sandboxed shell command (ls, grep, git status only)

### Routing & Self-Assessment
- `my_role()`: Get your current role, tier, capabilities, and what you can delegate to
- `route_advice(task_description)`: Get MemRL routing recommendation with Q-values
- `delegate(prompt, target_role, reason)`: Delegate to a specific role with outcome tracking
- `escalate(reason, target_role=None)`: Request escalation (up-chain or to specific role)
- `recall(query)`: Search episodic memory — returns Q-values and past routing outcomes

### LLM Delegation (low-level, no tracking)
- `llm_call(prompt, role='worker')`: Raw sub-LM call
- `llm_batch(prompts, role='worker')`: Parallel raw sub-LM calls

### Long Context Exploration
- `context_len()`: Return character count of context
- `chunk_context(n_chunks=4, overlap=200)`: Split context into N chunks with metadata
- `summarize_chunks(task, n_chunks=4, role='worker_general')`: Chunk + parallel worker summaries

### Tool Invocation
- `TOOL(tool_name, **kwargs)`: Invoke a registered tool, returns raw Python object
- `CALL(tool_name, **kwargs)`: Invoke a registered tool, returns JSON string (simpler)
  Example: `result = CALL("search_arxiv", query="transformers"); data = json.loads(result)`
- `list_tools()`: List available tools for your role

### Completion
- `FINAL(answer)`: Signal completion with the final answer (REQUIRED)"""

# Default rules for Root LM
DEFAULT_ROOT_LM_RULES = """## CRITICAL
1. **NO IMPORTS** - import/from are BLOCKED. The `json` module is pre-loaded, just use `json.loads()` directly.
2. **USE list_dir()** for files - NOT os.listdir or pathlib
3. **ALWAYS call FINAL(answer)** when you have your answer — this is REQUIRED to complete the task.
   Do NOT keep calling tools after you have enough information.

## Examples
List files: `result = list_dir('/path'); FINAL(result)`
Read file: `text = peek(1000, file_path='/path'); FINAL(text)`
Summarize PDF: `doc = json.loads(ocr_document('/path.pdf')); summary = llm_call(f"Summarize: {doc['full_text'][:6000]}", role='worker'); FINAL(summary)`

## Routing (OPTIONAL — only for complex multi-model tasks)
4. Simple tasks: just answer directly — do NOT call my_role() or route_advice() first
5. Only call my_role() if genuinely unsure about your capabilities
6. Only call route_advice() before delegating complex subtasks to specialists
7. Use `delegate()` over `llm_call()` when making a conscious routing choice
8. Call `escalate(reason)` if the task exceeds your tier — don't guess

## Other Rules
9. NEVER send full context to llm_call - use peek() or grep() first
10. Output only valid Python code - no markdown, no explanations"""


# ── ReAct Tool Loop Constants ──────────────────────────────────────────────

# Read-only tools safe for ReAct mode (no shell, no filesystem writes)
REACT_TOOL_WHITELIST = frozenset(
    {
        "web_search",
        "search_arxiv",
        "search_papers",
        "search_wikipedia",
        "get_wikipedia_article",
        "search_books",
        "calculate",
        "python_eval",
        "get_current_date",
        "get_current_time",
        "json_query",
        "fetch_wikipedia",
        # File tools — read-only access for investigation
        "read_file",
        "list_directory",
    }
)

# Vision-aware ReAct whitelist: standard tools + OCR extraction
VISION_REACT_TOOL_WHITELIST = REACT_TOOL_WHITELIST | frozenset({"ocr_extract"})

# Tools that _execute_vision_tool() actually handles — single source of truth
VISION_REACT_EXECUTABLE_TOOLS: frozenset[str] = frozenset(
    {
        "ocr_extract",
        "calculate",
        "get_current_date",
        "get_current_time",
    }
)

# Tool descriptions for the vision ReAct system prompt
VISION_TOOL_DESCRIPTIONS: dict[str, str] = {
    "ocr_extract": (
        'ocr_extract(image_base64="..."): Extract text from the image using OCR. '
        'The image is already loaded — pass image_base64="current" to use it.'
    ),
    "calculate": 'calculate(expression="..."): Evaluate a math expression',
    "get_current_date": "get_current_date(): Get today's date",
    "get_current_time": "get_current_time(): Get current time",
}

# ReAct format instructions
REACT_FORMAT = """You have access to the following tools:
{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: reason about what to do next
Action: tool_name(arg1="value1", arg2="value2")
Observation: the result of the action
... (repeat Thought/Action/Observation as needed, up to {max_turns} times)
Thought: I now know the final answer
Final Answer: the final answer to the original question

Important rules:
- Always start with a Thought before an Action
- Action arguments use keyword=value syntax (strings in quotes, numbers bare)
- After each Observation, decide if you have enough info for a Final Answer
- If no tools are needed, skip directly to Final Answer
- Be concise in your Final Answer — answer the question directly"""


def build_react_prompt(
    prompt: str,
    context: str = "",
    tool_registry: "Any | None" = None,
    max_turns: int = 5,
    tool_whitelist: frozenset[str] | None = None,
) -> str:
    """Build a ReAct-style prompt with tool descriptions.

    Args:
        prompt: The user's question.
        context: Optional context text.
        tool_registry: Optional tool registry for dynamic tool descriptions.
        max_turns: Maximum number of Thought/Action/Observation cycles.
        tool_whitelist: Optional override for REACT_TOOL_WHITELIST.
            If None, uses the module-level default.

    Returns:
        Formatted ReAct prompt string.
    """
    whitelist = tool_whitelist if tool_whitelist is not None else REACT_TOOL_WHITELIST

    # Build tool descriptions from whitelist
    tool_descriptions = []
    if tool_registry is not None:
        for tool_info in tool_registry.list_tools():
            name = tool_info.get("name", "")
            if name in whitelist:
                desc = tool_info.get("description", "No description")
                params = tool_info.get("parameters", {})
                param_strs = []
                for pname, pinfo in params.items():
                    ptype = pinfo.get("type", "string")
                    required = pinfo.get("required", False)
                    req_mark = " (required)" if required else ""
                    param_strs.append(f"  {pname}: {ptype}{req_mark}")
                param_block = "\n".join(param_strs) if param_strs else "  (no parameters)"
                tool_descriptions.append(f"- {name}: {desc}\n{param_block}")
    else:
        # Static fallback descriptions for common tools
        tool_descriptions = [
            '- calculate(expression="..."): Evaluate a math expression',
            "- get_current_date(): Get today's date",
            "- get_current_time(): Get current time",
            '- web_search(query="..."): Search the web',
            '- search_arxiv(query="...", max_results=5): Search arXiv papers',
            '- search_wikipedia(query="..."): Search Wikipedia articles',
            '- get_wikipedia_article(title="..."): Get full Wikipedia article',
            '- python_eval(code="..."): Evaluate Python expression safely',
            '- json_query(data="...", query="..."): Query JSON data with JMESPath',
            '- read_file(path="..."): Read file contents (text files only)',
            '- list_directory(path="..."): List directory contents',
        ]

    tool_desc_str = "\n".join(tool_descriptions)
    react_prompt = REACT_FORMAT.format(
        tool_descriptions=tool_desc_str,
        max_turns=max_turns,
    )

    if context:
        return f"{react_prompt}\n\nContext:\n{context}\n\nQuestion: {prompt}"
    return f"{react_prompt}\n\nQuestion: {prompt}"
