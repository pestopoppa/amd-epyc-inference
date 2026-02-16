"""Project-wide named constants.

Centralizes magic numbers that appear across multiple modules.
Log-only truncations (debug/info messages) are intentionally left as literals
since they have no semantic meaning beyond presentation.
"""

# -- TaskIR / Prompt Preview ---------------------------------------------------
# Maximum characters of a prompt/objective to embed in a TaskIR dict.
# Used by chat routing, streaming, proactive delegation, and review.
TASK_IR_OBJECTIVE_LEN = 200

# -- Answer Matching -----------------------------------------------------------
# Characters of tool output to compare against the answer text when deciding
# whether the REPL output was incorporated into the final answer.
TOOL_OUTPUT_MATCH_LEN = 200

# -- Delegation Brief Cache Key ------------------------------------------------
# Characters of the delegation brief used as a cache/dedup key.
DELEGATION_BRIEF_KEY_LEN = 200
