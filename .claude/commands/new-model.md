# New Model Onboarding

Register and/or benchmark a newly downloaded model.

**Model path:** $ARGUMENTS

## Your Task

1. **Parse the model path** - if doesn't start with `/`, prepend `/mnt/raid0/llm/lmstudio/models/`

2. **Verify file exists** - check the model file is present

3. **Auto-detect architecture** from filename:
   - "Qwen3" + ("A3B"|"A22B"|"A35B") but NOT "Next" → `qwen3moe`
   - "Qwen3" + "Next" → `qwen3next`
   - "GLM" + ("A32B"|MoE indicator) → `glm4moe`
   - "Mixtral" → `mixtral`
   - "DeepSeek" + "MoE" or "v2"|"v3" with large expert count → `deepseek2`
   - Everything else (Llama, Qwen2.5, DeepSeek-R1-Distill, etc.) → `dense`

4. **Generate short model name** from filename (strip quantization suffix, GGUF, etc.)

5. **Check if overnight benchmark is running**:
   ```bash
   pgrep -f "run_overnight_benchmark_suite" > /dev/null
   ```

6. **Based on status, do ONE of:**

   **A) If overnight benchmark IS running:**
   - Run: `./scripts/benchmark/add_model_to_benchmark.sh PATH NAME ARCH`
   - Report: "Model queued for running benchmark"

   **B) If overnight benchmark is NOT running:**
   - Add model to `orchestration/model_registry.yaml` under appropriate role
   - Ask user: "Run individual benchmark now, or save for next overnight run?"
   - If now: Run `./scripts/benchmark/run_thinking_rubric.sh PATH NAME ARCH` (or appropriate suite)

7. **Report result** with model name, architecture detected, and action taken

## Example Usage
```
/new-model tensorblock/Qwen2.5-Math-1.5B-Instruct-GGUF/Qwen2.5-Math-1.5B-Instruct-Q6_K.gguf
/new-model /mnt/raid0/llm/models/CustomModel.gguf
```
