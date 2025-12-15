# Agent Definitions

Specialized agents for the AMD EPYC inference optimization project. Model selection is **task-based**, not agent-based.

## Primary Agents

| Agent | File | Default Model | Role |
|-------|------|---------------|------|
| **Lead Developer** | `lead-developer.md` | Sonnet | Architecture, coordination, strategic decisions |
| **Research Engineer** | `research-engineer.md` | Sonnet | C++ implementation, debugging, novel approaches |
| **Research Writer** | `research-writer.md` | Sonnet | Report synthesis, documentation, literature review |
| **Benchmark Analyst** | `benchmark-analyst.md` | Haiku | Benchmark execution, data collection, metrics |

## Support Agents

| Agent | File | Default Model | Role |
|-------|------|---------------|------|
| **Sysadmin** | `sysadmin.md` | Sonnet | System configuration, NUMA, CPU governor |
| **Build Engineer** | `build-engineer.md` | Sonnet | CMake, compiler flags, build issues |
| **Model Engineer** | `model-engineer.md` | Sonnet | GGUF conversion, quantization |
| **Safety Reviewer** | `safety-reviewer.md` | Opus | Risk assessment, security review |

## Task-Based Model Selection

Models are selected based on **task complexity**, not which agent is invoked:

```
TASK COMPLEXITY → MODEL SELECTION

Novel/Complex (Opus):
  - "Design new speculation approach"
  - "Debug 0% acceptance rate"
  - "Implement MoE self-drafting"
  - "Analyze architectural incompatibility"

Research/Synthesis (Sonnet):
  - "Find MoE code in llama.cpp"
  - "Update report with K-tuning results"
  - "Compare K=8,16,24 and summarize"
  - "Add CLI flag for new parameter"

Routine/Execution (Haiku):
  - "Run bench_zen5.sh"
  - "Parse CSV results"
  - "Check if model file exists"
  - "Build with cmake"
```

## Decision Flow

```
┌─────────────────────────────────────────────────────────┐
│                    TASK ARRIVES                         │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  Is this novel or    │
              │  complex debugging?  │
              └──────────┬───────────┘
                    YES  │  NO
                    ┌────┴────┐
                    ▼         ▼
               ┌───────┐  ┌────────────────────┐
               │ OPUS  │  │ Is this research,  │
               └───────┘  │ synthesis, or      │
                          │ straightforward    │
                          │ coding?            │
                          └─────────┬──────────┘
                              YES   │  NO
                              ┌─────┴─────┐
                              ▼           ▼
                         ┌────────┐  ┌───────┐
                         │ SONNET │  │ HAIKU │
                         └────────┘  └───────┘
```

## Invoking Agents

Reference agents in prompts using `@agent-name`:

```
@lead-developer evaluate whether we should implement Track 7 CAS-Spec
@research-engineer implement MoE Top-1 gating modification
@research-writer update the research report with K-tuning results
@benchmark-analyst run systematic optimization benchmark on all models
```

## Agent Coordination Flow

```
                    ┌─────────────────┐
                    │ Lead Developer  │
                    │  (coordinates)  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│    Research     │ │   Benchmark     │ │    Research     │
│    Engineer     │ │    Analyst      │ │     Writer      │
│  (implements)   │ │   (measures)    │ │  (documents)    │
└─────────────────┘ └─────────────────┘ └─────────────────┘
         │                   │                   │
         └───────────────────┴───────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  Support Agents │
                    │  (as needed)    │
                    └─────────────────┘
```

## Model Selection Summary

| Task Type | Model | Cost | Speed |
|-----------|-------|------|-------|
| Novel design, complex debugging, architecture | **Opus** | High | Slow |
| Research, synthesis, routine coding | **Sonnet** | Medium | Fast |
| Benchmark execution, log parsing, status checks | **Haiku** | Low | Fastest |

**Rule:** Start with the cheapest model that can handle the task. Escalate if blocked.
