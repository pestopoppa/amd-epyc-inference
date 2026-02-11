context: str — full input text (use peek/grep to inspect, don't pass to llm_call)
artifacts: dict — store results between turns
peek(n, file_path=None) → first n chars of context/file
grep(pattern, file_path=None) → regex matches in context/file
llm_call(prompt, role='worker') → sub-LM call (keep prompt short)
escalate(reason, target_role=None) → hand off to higher tier
FINAL(value) → signal task completion with your computed result (REQUIRED)
CALL(name, **kw) → invoke any registered tool, returns JSON string
list_tools() → discover ALL available tools (web, files, research, code quality, etc.)
