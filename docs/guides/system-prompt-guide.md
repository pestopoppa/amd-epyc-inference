# How to Customize Claude Code's System Prompt

There are several ways to inject context and instructions into Claude Code, ranging from simple to advanced.

---

## 1. CLAUDE.md Files (Recommended Starting Point)

Claude Code automatically reads `CLAUDE.md` files and includes them in context.

### Locations (in order of precedence):
```
~/.claude/CLAUDE.md          # User-level (applies to all projects)
/path/to/project/CLAUDE.md   # Project root (most common)
/path/to/project/.claude/CLAUDE.md  # Alternative project location
```

### Usage:
Simply place the file and start Claude Code in that directory:
```bash
cd ~/your-project
claude
```

Claude automatically picks it up — no flags needed.

### Best for:
- Project-specific context and conventions
- Build instructions, coding standards
- Domain knowledge that should persist across sessions

---

## 2. --append-system-prompt Flag

Append additional instructions to the system prompt at runtime.

### Usage:
```bash
claude --append-system-prompt "Focus on performance optimization. Always consider memory bandwidth constraints. Prefer solutions that minimize TLB misses."
```

### From a file:
```bash
claude --append-system-prompt "$(cat my-instructions.txt)"
```

### Best for:
- Session-specific instructions
- Temporary focus areas
- Quick experiments with different prompting strategies

---

## 3. --system-prompt-file Flag

Load a complete custom system prompt from a file (more control than append).

### Usage:
```bash
claude --system-prompt-file ~/prompts/optimization-expert.txt
```

### Best for:
- Completely custom personas or workflows
- Switching between different "modes" of operation
- Team-standardized prompt configurations

---

## 4. Custom Slash Commands

Create reusable commands that inject specific context or workflows.

### Location:
```
.claude/commands/optimize.md
```

### Example content (.claude/commands/benchmark.md):
```markdown
# Benchmark Configuration Assistant

You are helping configure and run benchmarks on an AMD EPYC 9655 system.

Before suggesting any benchmark:
1. Confirm the model file path exists
2. Check current CPU governor setting
3. Verify NUMA configuration
4. Ensure OMP_NUM_THREADS=1 is set

Always output benchmark commands with full environment setup.
```

### Usage in Claude Code:
```
/benchmark
```

### Best for:
- Workflow-specific contexts
- Team-shared procedures
- Complex multi-step operations

---

## 5. Role Prompts

Use repository role prompts for specialist behavior.

### Location:
```
agents/
```

### Structure:
- `agents/AGENT_INSTRUCTIONS.md` (execution contract)
- `agents/shared/*.md` (shared policy)
- `agents/*.md` (lean role overlays)

### Best for:
- Consistent role behavior across sessions
- Shared policy without duplication
- Stable handoff contracts by role

---

## 6. Hooks (Advanced)

Automatically inject context or run scripts at specific points.

### Location:
```
.claude/settings.json
```

### Example — PreToolUse hooks:
```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "bash \"$CLAUDE_PROJECT_DIR/scripts/hooks/check_pytest_safety.sh\""
          }
        ]
      }
    ]
  }
}
```

### Example script (`scripts/hooks/check_pytest_safety.sh`):
```bash
#!/bin/bash
set -euo pipefail
INPUT=$(cat)
CMD=$(echo "$INPUT" | jq -r '.tool_input.command // empty')
echo "$CMD" | grep -qP 'pytest.*-n\s*auto' && exit 2 || exit 0
```

### Best for:
- Dynamic context injection
- Pre/post task automation
- Guardrails and validation

---

## 7. Environment Variables

Some behaviors can be controlled via environment:

```bash
# Increase context window awareness
export CLAUDE_CODE_MAX_CONTEXT=200000

# Enable verbose MCP logging
export CLAUDE_MCP_DEBUG=1
```

---

## Recommended Setup for Your Use Case

For your AMD EPYC 9655 system, I recommend this structure:

```
/mnt/raid0/llm/
├── CLAUDE.md                      # Main project context — COPY HERE
├── .claude/
│   ├── commands/
│   │   ├── benchmark.md           # /benchmark command
│   │   ├── audit.md               # /audit command
│   │   └── configure.md           # /configure command
│   ├── skills/                    # Packaged local skills (SKILL.md format)
│   └── settings.json              # Hook configuration
├── agents/                        # Role prompts + shared agent policy
│   ├── AGENT_INSTRUCTIONS.md
│   ├── shared/
│   └── *.md
├── UTILS/
│   ├── system_audit.sh            # Pre-change state capture
│   ├── bench_zen5.sh              # Systematic benchmarking
│   └── run_inference.sh           # Optimized inference wrapper
├── LOGS/                          # All benchmark and audit logs
├── llama.cpp/                     # Inference engine
├── hf/                            # HuggingFace models
└── models/                        # Converted GGUF models
```

### Quick Start:
```bash
# 1. Copy CLAUDE.md to your project root
cp CLAUDE.md /mnt/raid0/llm/

# 2. Copy utility scripts
mkdir -p /mnt/raid0/llm/UTILS
cp system_audit.sh bench_zen5.sh run_inference.sh /mnt/raid0/llm/UTILS/
chmod +x /mnt/raid0/llm/UTILS/*.sh

# 3. Run Claude Code from that directory
cd /mnt/raid0/llm
claude

# 4. First task: run system audit
# (Claude will know to use /mnt/raid0/llm/UTILS/system_audit.sh)
```

### Session-Specific Focus:
```bash
# Focus on speculative decoding tuning
claude --append-system-prompt "This session: focus on speculative decoding optimization. Target 50+ t/s on DeepSeek-R1-32B."

# Focus on compilation and build optimization
claude --append-system-prompt "This session: rebuild llama.cpp with optimal Zen 5 flags. Verify AVX-512 VNNI/VBMI are enabled."

# Focus on benchmarking
claude --append-system-prompt "This session: run systematic benchmarks across thread counts. Find optimal configuration."
```

---

## Tips

- **CLAUDE.md is persistent** — great for things Claude should always know
- **--append-system-prompt is ephemeral** — great for session-specific focus
- **Slash commands are invokable** — great for workflows you repeat
- **Subagents are delegatable** — great for specialized expertise
- **Keep CLAUDE.md under ~4000 tokens** — too long and it dilutes focus
