# Understanding CLAUDE.md

This guide helps human readers navigate CLAUDE.md - the AI context file for this project.

## What is CLAUDE.md?

CLAUDE.md is a context file that Claude Code reads at session start. It provides:
- Project-specific rules and constraints
- Hardware and system specifications
- Workflow procedures
- Quick reference commands

**It is NOT documentation for humans** - it's optimized for AI parsing. This guide explains its structure.

## Why Is It Organized This Way?

### Monolithic Design

Claude Code loads CLAUDE.md entirely into context at session start. A single file:
- Ensures all context loads together
- Avoids partial reads or missing context
- Enables cross-referencing within the document

### Tables and Code Blocks

AI models parse structured formats better than prose:
- **Tables**: Quick lookup of model/command mappings
- **Code blocks**: Copy-paste ready commands
- **Headers**: Section navigation

### Repetition

Some information appears multiple times in different forms:
- Table (quick reference)
- Prose (context)
- Commands (action)

This is intentional - different query types need different formats.

## Section Guide

| Section | Purpose | Update Frequency |
|---------|---------|------------------|
| **Critical Constraints** | Filesystem rules | Rarely |
| **System Identity** | Host/user info | Never |
| **Hardware Specifications** | CPU/RAM specs | Never |
| **Current Status** | Best results, production tracks | Per benchmark |
| **Hierarchical Orchestration** | Agent tiers, philosophy | Rarely |
| **Directory Structure** | Project layout | Per restructure |
| **Session Startup** | Required commands | Rarely |
| **Quick Reference Commands** | Inference commands | Per optimization |
| **Orchestration Workflow** | TaskIR, routing | Per design change |
| **Verification Gates** | Gate order | Rarely |
| **Model Routing** | Tier selection | Per benchmark |
| **Logging Requirements** | Log patterns | Rarely |
| **Model Testing Workflow** | New model process | Per discovery |
| **Benchmarking Pitfalls** | Common mistakes | Per discovery |
| **Claude-as-Judge** | Scoring rubric | Per hardening |
| **Benchmark Hardening** | Suite changes | Per hardening |
| **Key Resources** | Document links | Per restructure |

## Finding Information

### "How do I run inference?"

→ **Quick Reference Commands** section

### "What model should I use?"

→ **Model Routing Strategy** section, or **Hierarchical Orchestration** for role definitions

### "What are the rules?"

→ **Critical Constraints** (filesystem), **Verification Gates** (quality)

### "Something broke, what's the workaround?"

→ **Benchmarking Pitfalls**, or [docs/reference/models/QUIRKS.md](docs/reference/models/QUIRKS.md)

### "What's the current best configuration?"

→ **Current Status** section (Best Results table)

## Human-Friendly Alternatives

For human reading, these documents are better organized:

| Topic | Read This Instead |
|-------|-------------------|
| Research journey | [docs/chapters/INDEX.md](docs/chapters/INDEX.md) |
| Model reference | [docs/reference/models/MODELS.md](docs/reference/models/MODELS.md) |
| Commands | [docs/reference/commands/QUICK_REFERENCE.md](docs/reference/commands/QUICK_REFERENCE.md) |
| Benchmark results | [docs/reference/benchmarks/RESULTS.md](docs/reference/benchmarks/RESULTS.md) |
| Getting started | [docs/guides/getting-started.md](docs/guides/getting-started.md) |

## How Claude Uses CLAUDE.md

1. **Session start**: Full file loaded into context
2. **Task routing**: Checks Model Routing section
3. **Command execution**: Copies from Quick Reference
4. **Constraint checking**: References Critical Constraints
5. **Error handling**: Checks Benchmarking Pitfalls

## Updating CLAUDE.md

If you need to update CLAUDE.md:

1. **Add to existing sections** rather than creating new ones
2. **Use tables** for collections of items
3. **Include commands** with full paths
4. **Keep sections focused** - CLAUDE.md is already long
5. **Test with Claude Code** to verify AI can find new content

## Size and Performance

CLAUDE.md is approximately 750 lines / 26KB. This is large but acceptable because:
- Claude Code has sufficient context window
- All information loads in one operation
- Cross-referencing works within the document

If CLAUDE.md exceeds 1000 lines, consider extracting to reference docs.

---

*This guide is for human orientation. For actual work, use the structured documents in [docs/](docs/).*
