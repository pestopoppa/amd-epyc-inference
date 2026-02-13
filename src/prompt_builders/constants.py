"""Prompt constants: tool descriptions, rules, ReAct format, vision whitelists."""

from __future__ import annotations

from typing import Any


# Default tool descriptions for Root LM
# Following Claude Code pattern: each tool has "when to use" AND "when NOT to use"
DEFAULT_ROOT_LM_TOOLS = """### Context & Files
- `context`: str - The full input context. Do NOT send to LLM calls directly (too large).
- `artifacts`: dict - Store intermediate results between turns.
- `peek(n, file_path=None)`: Return first n characters of context or file. Use when you
  need to inspect content. Do NOT use if you can answer from knowledge alone.
- `grep(pattern, file_path=None)`: Search context or file with regex. Use for finding
  specific patterns in large text. Do NOT use for simple keyword questions.
- `list_dir(path)`: List directory contents as JSON. Use for file exploration tasks.
  Do NOT use for questions unrelated to files.
- `file_info(path)`: Get file metadata (size, type, modified date). Use when file
  properties matter. Do NOT use if you just need file contents.

### Document Processing (return JSON - use json.loads())
- `ocr_document(path)`: Extract text from PDF. Use ONLY for PDF/image files that need
  OCR. Do NOT use for plain text files or questions answerable without documents.
- `analyze_figure(image_path, prompt)`: Analyze image with vision model. Use for charts,
  diagrams, photos. Do NOT use for text-only tasks.
- `extract_figure(pdf_path, page, bbox)`: Crop figure from PDF. Use when you need a
  specific region. Do NOT use if ocr_document gives sufficient text.

### Web & Shell
- `web_search(query, max_results=5)`: Search the web via DuckDuckGo. Use ONLY when you
  need current/real-time info, or facts beyond your training data. Do NOT use for
  general knowledge, reasoning, or multiple-choice questions you can answer directly.
- `fetch_docs(url)`: Fetch content from a URL. Use when you have a specific URL to read.
  Do NOT use for general searches (use web_search instead).
- `run_shell(cmd)`: Run sandboxed shell command (ls, grep, git status only). Use for
  system exploration. Do NOT use for file reading (use peek instead).

### Knowledge Retrieval (via CALL)
- `search_arxiv(query, max_results=10)`: Search arXiv papers. Use for academic/research
  questions requiring recent publications. Do NOT use for basic science questions.
- `search_papers(query, max_results=10)`: Search Semantic Scholar with citation counts.
  Use when citation metrics matter. Do NOT use for simple factual questions.
- `search_wikipedia(query, max_results=5)`: Search Wikipedia. Use when you need verified
  encyclopedic info. Do NOT use for questions you can answer from knowledge.
- `get_wikipedia_article(title)`: Get full Wikipedia article. Use after search_wikipedia
  identifies a relevant article. Do NOT use without knowing the exact title.
- `search_books(query, max_results=10)`: Search Google Books. Use for book-specific
  queries. Do NOT use for general knowledge questions.

### Code Quality (via CALL)
- `run_tests(test_path, test_pattern=None)`: Run pytest tests. Use when testing code
  changes. Do NOT use for non-code tasks.
- `lint_python(file_path, fix=False)`: Lint Python with ruff. Use for code quality
  checks. Do NOT use for non-Python or non-code tasks.
- `json_parse(content, extract_path=None)`: Parse/validate JSON. Use for structured
  data extraction. Do NOT use if data is already parsed.

### Routing & Self-Assessment
- `my_role()`: Get your role, tier, capabilities. Use ONLY if genuinely unsure about
  your capabilities. Do NOT call on every task.
- `route_advice(task_description)`: Get MemRL routing recommendation. Use ONLY before
  delegating complex subtasks. Do NOT use for simple tasks.
- `delegate(prompt, target_role, reason)`: Delegate to specialist with tracking. Use
  for complex subtasks needing different expertise. Do NOT use for simple tasks.
- `escalate(reason, target_role=None)`: Request escalation. Use when task exceeds your
  tier. Do NOT use if you can complete the task yourself.
- `recall(query)`: Search episodic memory for past outcomes. Use for routing decisions.
  Do NOT use for answering user questions.

### LLM Delegation (low-level, no tracking)
- `llm_call(prompt, role='worker')`: Raw sub-LM call. Use for subtasks needing LLM.
  Do NOT send full context (use peek/grep first). Do NOT use for simple questions.
- `llm_batch(prompts, role='worker')`: Parallel sub-LM calls. Use for multiple
  independent subtasks. Do NOT use for single questions.

### Long Context Exploration
- `context_len()`: Return character count. Use to check if context is large.
- `chunk_context(n_chunks=4, overlap=200)`: Split context into chunks. Use for
  long document processing. Do NOT use for short contexts.
- `summarize_chunks(task, n_chunks=4, role='worker_general')`: Chunk + parallel
  summaries. Use for long document summarization. Do NOT use for short texts.

### Tool Invocation
- `TOOL(tool_name, **kwargs)`: Invoke registered tool, returns Python object.
- `CALL(tool_name, **kwargs)`: Invoke registered tool, returns JSON string.
  Example: `result = CALL("search_arxiv", query="transformers"); data = json.loads(result)`
- `list_tools()`: Discover ALL available tools. Use to find specialized tools.

### Completion
- `FINAL(answer)`: Signal completion with the final answer. REQUIRED for every task."""

# Compact tool descriptions for MINIMAL prompt style (~140 tokens vs ~1450)
# Core tools only; model calls list_tools() when it needs extras.
COMPACT_ROOT_LM_TOOLS = """\
context: str — full input text (use peek/grep to inspect, don't pass to llm_call)
artifacts: dict — store results between turns
peek(n, file_path=None) → first n chars of context/file
grep(pattern, file_path=None) → regex matches in context/file
file_write_safe(path, content) → write code to /mnt/raid0/llm/tmp/ for iterative editing
llm_call(prompt, role='worker') → sub-LM call (keep prompt short)
escalate(reason, target_role=None) → hand off to higher tier
FINAL(answer) → signal task completion (REQUIRED for every task)
CALL(name, **kw) → invoke any registered tool, returns JSON string
list_tools() → discover ALL available tools (web, files, research, code quality, etc.)"""

# Default rules for Root LM
DEFAULT_ROOT_LM_RULES = """## WHEN TO USE TOOLS vs DIRECT ANSWER
- **Answer directly** for: factual lookups, multiple-choice, short math
- **Reason thoroughly** for: explanations, analysis, multi-step problems, "why" questions
- **Use tools** for: file access, web search, current events, running code, document processing
- **Match depth to request**: concise for simple questions, detailed for complex ones

## CRITICAL RULES
1. **SAFE IMPORTS ONLY** - `math`, `json`, `re`, `numpy`, `scipy`, `itertools`, `collections`, `functools`,
   `statistics`, `datetime`, `fractions`, `decimal` are available. `os`, `sys`, `subprocess`, `socket` are BLOCKED.
2. **USE list_dir()** for files - NOT os.listdir or pathlib
3. **ALWAYS call FINAL(answer)** to complete the task. Do NOT keep calling tools after
   you have enough information.

## EXAMPLES: Direct Answer (NO tools needed)
Factual: `FINAL("Paris")`  # "What is the capital of France?"
Multiple choice: `FINAL("B")`  # "Which option is correct? A) ... B) ..."
Short math: `FINAL("42")`  # "What is 6 * 7?"
Reasoning: `FINAL("Step 1: The premises state all A are B and all B are C. Step 2: By transitivity, all A are C. Step 3: Since x is A, x must be C. Therefore x is C.")`  # "Explain why x is C given..."
Analysis: `FINAL("The function has O(n log n) complexity because the outer loop runs n times and the inner binary search runs log n times. This is optimal for comparison-based sorting.")`  # "Analyze the complexity"

## EXAMPLES: Tool Use (external data needed)
List files: `result = list_dir('/path'); FINAL(result)`
Read file: `text = peek(1000, file_path='/path'); FINAL(text)`
Current info: `results = CALL("web_search", query="2024 election results"); FINAL(json.loads(results))`
Research: `results = CALL("search_arxiv", query="speculative decoding"); FINAL(json.loads(results))`
Run tests: `results = CALL("run_tests", test_path="tests/"); FINAL(json.loads(results))`
Summarize PDF: `doc = json.loads(ocr_document('/path.pdf')); FINAL(doc['full_text'][:2000])`

## EXAMPLES: Competitive Programming (stdin/stdout: USACO, Codeforces, etc.)
NOTE: input() is BLOCKED in the REPL. Wrap code in a string, test with CALL, submit with FINAL.
```
solution = '''
import sys
input = sys.stdin.readline
n = int(input())
a = list(map(int, input().split()))
best = cur = a[0]
for x in a[1:]:
    cur = max(x, cur + x)
    best = max(best, cur)
print(best)
'''
test_out = CALL("run_python_code", code=solution, stdin_data="5\n-2 1 -3 4 -1")
# verify output looks correct, then submit the code itself
FINAL(solution)
```

## EXAMPLES: Write/Fix a Function (LeetCode, DebugBench, etc.)
When the task says "write a function" or "fix the bug", submit the CODE ITSELF as a string.
```
solution = '''
def shortestPalindrome(s: str) -> str:
    if not s: return s
    combined = s + "#" + s[::-1]
    lps = [0] * len(combined)
    length = 0
    for i in range(1, len(combined)):
        while length and combined[i] != combined[length]:
            length = lps[length - 1]
        if combined[i] == combined[length]:
            length += 1
        lps[i] = length
    return s[lps[-1]:][::-1] + s
'''
FINAL(solution)
```

## COMPLEX CODE (algorithms, implementations)
- Write code to file: `file_write_safe("/mnt/raid0/llm/tmp/solution.py", code)`
- Test it: `exec(open("/mnt/raid0/llm/tmp/solution.py").read())` with sample input
- Edit incrementally — read, modify, rewrite. Do NOT regenerate from scratch.
- If stuck after 2 attempts: consult architect or escalate to coder_escalation.
- For stdin/stdout programs: wrap in string, use `CALL("run_python_code", code=..., stdin_data=...)` to test. Do NOT use `import sys` or `input()` directly — they are blocked.

## ESCALATION (three modes)
- **Consult**: `answer = llm_call("Be concise. " + question, role="architect")` then `FINAL(answer)`.
  Ask a stronger model for help — you keep control and format the answer.
  Example: `answer = llm_call("Answer with just the letter. " + question, role="architect"); FINAL(answer)`
- **Delegate**: `escalate(reason, target_role="coder_escalation")` — hand off code tasks to a specialist coder.
  Use for: algorithms, competitive programming, complex implementations.
- **Handoff**: `escalate(reason)` — transfer the entire task when it exceeds your tier.

## OTHER RULES
- NEVER send full context to llm_call - use peek() or grep() first
- Output only valid Python code - no markdown, no explanations around the code
- Do NOT reason in Python comments. Think before writing code, then write only executable statements ending with FINAL()."""


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
