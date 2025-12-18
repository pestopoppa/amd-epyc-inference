# New Model Onboarding

Register a new model with VERIFIED configuration. Uses registry as single source of truth.

**Model path:** $ARGUMENTS

## Your Task

Run the Python onboarding module and guide the user through confirming the results.

### Step 1: Run Onboarding

```bash
python3 /mnt/raid0/llm/claude/scripts/lib/onboard.py "$ARGUMENTS"
```

This will:
1. Validate the model path exists
2. Detect architecture, family, quantization from filename
3. Generate applicable optimization configs
4. Find compatible drafts (or generate compatible_targets if this is a draft)
5. Run health check to verify model launches
6. Suggest candidate roles based on model properties

### Step 2: Review Results with User

If onboarding succeeds, present the results:

**Model Information:**
- Name: [extracted short name]
- Architecture: [dense/moe/qwen3moe/ssm_moe_hybrid/etc.]
- Family: [Qwen3/Qwen2.5/Llama/etc.]
- Size: [X.X GB]
- Tier: [A/B/C/D]

**Optimization Configs:** [X configs]
- baseline: 1
- moe: X (if MoE architecture)
- spec: X (if dense with compatible drafts)
- lookup: X (if applicable)

**Compatible Drafts:** [list or "None - this IS a draft model"]

**Health Check:**
- Status: [PASSED/FAILED]
- Speed: [X.X t/s]
- Flags needed: [list or "none"]

**Suggested Roles:** [list]
**Suggested Role Name:** [role_name]

### Step 3: Confirm Candidate Roles

Ask the user:
> "The suggested candidate roles are: [roles]. Do you want to modify these?"

Options:
- Keep as suggested
- User specifies different roles

### Step 4: Push to Registry

After user confirms, add the entry:

```python
import sys
sys.path.insert(0, "/mnt/raid0/llm/claude/scripts/lib")
from registry import load_registry

registry = load_registry()
registry.add_model_entry("ROLE_NAME", ENTRY_DICT)
print(f"Added {ROLE_NAME} to registry")
```

### Step 5: Verify Entry Works

Run validation to confirm the new entry works with all configs:

```bash
python3 << 'EOF'
import sys
sys.path.insert(0, "/mnt/raid0/llm/claude/scripts/lib")
from registry import load_registry
from executor import Executor

registry = load_registry()
executor = Executor(registry)

role = "NEW_ROLE_NAME"
arch = registry.get_architecture(role)
configs = executor.get_configs_for_architecture(arch, role, registry)
print(f"✓ {role}: {len(configs)} configs validated")
EOF
```

### Step 6: Offer Benchmark

Ask the user:
> "Run benchmark now?"

**If YES and this is a TARGET model:**
```bash
cd /mnt/raid0/llm/claude/scripts/benchmark
./run_benchmark.py --model NEW_ROLE_NAME
```

**If YES and this is a DRAFT model:**
Run benchmarks on all targets that can use this draft:
```bash
cd /mnt/raid0/llm/claude/scripts/benchmark
# Get targets from registry
python3 -c "
import sys
sys.path.insert(0, '../lib')
from registry import load_registry
registry = load_registry()
targets = registry.get_targets_for_draft('DRAFT_ROLE_NAME')
for t in targets:
    print(t)
" | while read target; do
    ./run_benchmark.py --model "$target"
done
```

## Error Handling

**If onboarding fails:**
- Model not found → suggest searching: `find /mnt/raid0/llm -name "*PATTERN*" -type f`
- Already in registry → show existing role name, ask if user wants to re-benchmark
- Health check failed → report the error, suggest manual investigation

**If health check needs special flags:**
The onboarding module automatically tries different flag combinations. If a model needs `--no-conversation` or `--jinja`, this is recorded and included in the registry entry.

## Example Usage

```
/new-model tensorblock/Qwen2.5-Math-1.5B-Instruct-GGUF/Qwen2.5-Math-1.5B-Instruct-Q6_K.gguf
/new-model /mnt/raid0/llm/models/CustomModel.gguf
```

## Why This Approach

1. **Single source of truth** - All logic in Python modules, registry is authoritative
2. **No hardcoded paths** - Binaries from `runtime_defaults.binaries`, drafts from `compatible_targets`
3. **Health check catches quirks** - Before wasting benchmark time
4. **User confirms roles** - No surprises in registry
5. **Immediate validation** - Verify configs work before benchmarking
