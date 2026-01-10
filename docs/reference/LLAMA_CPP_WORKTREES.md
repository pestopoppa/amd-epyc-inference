# llama.cpp Worktree Setup

## Overview

Multiple agents share access to llama.cpp. To prevent conflicts, we use **git worktrees** to isolate production and experimental work.

## Directory Layout

| Directory | Branch | Purpose |
|-----------|--------|---------|
| `/mnt/raid0/llm/llama.cpp` | `production-consolidated` | **Production** - benchmarks, stable inference |
| `/mnt/raid0/llm/llama.cpp-experimental` | `feature/*` branches | **Experimental** - new features, research |

## Rules

### Production Directory (`/mnt/raid0/llm/llama.cpp`)

- **NEVER** checkout a different branch here
- **NEVER** commit experimental work here
- Keep on `production-consolidated` at all times
- This is what benchmarks and production inference use

### Experimental Directory (`/mnt/raid0/llm/llama.cpp-experimental`)

- Use for all experimental/research work
- Switch branches freely within this directory
- Build experimental binaries here: `./build/bin/llama-*`
- Changes here don't affect production

## Common Operations

### Starting Experimental Work

```bash
cd /mnt/raid0/llm/llama.cpp-experimental

# Create new feature branch (from production base)
git checkout production-consolidated
git checkout -b feature/my-new-feature

# Or switch to existing feature branch
git checkout feature/paged-attention
```

### Building Experimental Version

```bash
cd /mnt/raid0/llm/llama.cpp-experimental
cmake -B build -DGGML_NATIVE=ON -DGGML_AVX512=ON ...
cmake --build build -j 96

# Run experimental binary
./build/bin/llama-cli --version
```

### Checking Current State

```bash
# See all worktrees
cd /mnt/raid0/llm/llama.cpp
git worktree list

# Expected output:
# /mnt/raid0/llm/llama.cpp               6b43356a1 [production-consolidated]
# /mnt/raid0/llm/llama.cpp-experimental  xxxxxxxx [feature/something]
```

### Adding Another Worktree (if needed)

```bash
cd /mnt/raid0/llm/llama.cpp
git worktree add /mnt/raid0/llm/llama.cpp-feature2 feature/another-branch
```

## Current Branches

| Branch | Status | Description |
|--------|--------|-------------|
| `production-consolidated` | **PRODUCTION** | Stable with SWA fixes, parallel repack |
| `feature/paged-attention` | Experimental | CPU paged attention implementation |
| `mtp-branch` | Research | Multi-token prediction |
| `feature/eagle-*` | Research | EAGLE speculation |

## Troubleshooting

### "I accidentally worked on production-consolidated"

1. Stash or commit your changes
2. Create a feature branch: `git checkout -b feature/my-work`
3. Switch production back:
   ```bash
   cd /mnt/raid0/llm/llama.cpp
   git checkout production-consolidated
   ```
4. Move your work to experimental worktree

### "Build is using wrong version"

Check which directory you're in:
```bash
pwd  # Should be /mnt/raid0/llm/llama.cpp for production
git branch --show-current  # Should be production-consolidated
./build/bin/llama-cli --version  # Verify commit hash
```

---

*Created: 2026-01-10 after agents conflicted on shared directory*
