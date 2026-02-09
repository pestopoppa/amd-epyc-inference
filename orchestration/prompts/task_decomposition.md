Decompose this task into 2-5 parallel-executable steps.
Return ONLY a JSON array, no markdown fences, no explanation.

Each step: {{"id":"S1","actor":"worker"|"coder"|"architect","action":"what to do","depends_on":[],"parallel_group":"group_name","outputs":["result"]}}

Rules:
- Independent steps share a parallel_group so they run simultaneously
- Use depends_on only when a step needs another step's output
- actor: "worker" for exploration/summarization, "coder" for code, "architect" for design
- Keep actions concise (1-2 sentences)

Task: {objective}{context_note}

JSON: