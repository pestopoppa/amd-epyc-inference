# Orchestration Routing Topology

> Auto-generated from `src/roles.py` and `src/escalation.py`
> Last updated: 2026-02-07

## Complete Routing Graph

```mermaid
flowchart TB
    subgraph Input["Request Entry"]
        REQ[/"User Request"/]
    end

    subgraph TierA["Tier A: Frontdoor"]
        FD[frontdoor]
    end

    subgraph Classification["Intent Classification"]
        ROUTE{{"HybridRouter /<br/>RuleBasedRouter"}}
    end

    subgraph TierC["Tier C: Workers (Parallel)"]
        WG[worker_general]
        WM[worker_math]
        WS[worker_summarize]
        WV[worker_vision]
        TR[toolrunner]
    end

    subgraph TierB["Tier B: Specialists"]
        CP[coder_escalation]
        CE[coder_escalation]
        ILC[ingest_long_context]
        THR[thinking_reasoning]
        AG[architect_general]
        AC[architect_coding]
    end

    subgraph TierD["Tier D: Draft Models"]
        DC[draft_coder]
        DG[draft_general]
    end

    subgraph MemRL["MemRL Advisory Layer"]
        EP[(EpisodicStore<br/>Q-values)]
        HG[(HypothesisGraph<br/>confidence)]
        FG[(FailureGraph<br/>anti-memory)]
        TPR[TwoPhaseRetriever]
    end

    subgraph Retrieval["Code Retrieval Layer (Phase 4: Dual Containers)"]
        NPC[("NextPLAID-code<br/>:8088")]
        NPD[("NextPLAID-docs<br/>:8089")]
        CI[(code index<br/>LateOn-Code-edge)]
        DI[(docs index<br/>answerai-colbert-sm)]
    end

    %% Entry flow
    REQ --> ROUTE
    ROUTE -->|"simple"| FD
    ROUTE -->|"code"| CP
    ROUTE -->|"ingest"| ILC
    ROUTE -->|"vision"| WV
    ROUTE -->|"parallel"| WG

    %% Frontdoor can handle directly or delegate
    FD -->|"delegate"| CP
    FD -->|"escalate"| CP

    %% Worker escalation (all workers → coder_escalation)
    WG -->|"escalate"| CP
    WM -->|"escalate"| CP
    WS -->|"escalate"| CP
    WV -->|"escalate"| CP
    TR -->|"escalate"| CP

    %% Coder escalation
    CP -->|"escalate"| AG
    CE -->|"escalate"| AC
    THR -->|"escalate"| AG

    %% Ingest escalation (direct to architect)
    ILC -->|"escalate"| AG

    %% Architects are terminal (no escalation)
    AG -->|"terminal"| END([Response])
    AC -->|"terminal"| END

    %% Draft models support spec decode (not escalation)
    DC -.->|"spec decode"| CP
    DC -.->|"spec decode"| CE
    DG -.->|"spec decode"| WG

    %% MemRL influences routing decisions
    EP --> TPR
    HG --> TPR
    FG --> TPR
    TPR -.->|"suggest target"| ROUTE
    TPR -.->|"suggest escalation"| CP
    TPR -.->|"suggest escalation"| WG

    %% Code retrieval (REPL code_search/doc_search) — Phase 4 dual containers
    FD -.->|"code_search()"| NPC
    CP -.->|"code_search()"| NPC
    WG -.->|"code_search()"| NPC
    FD -.->|"doc_search()"| NPD
    CP -.->|"doc_search()"| NPD
    WG -.->|"doc_search()"| NPD
    NPC --> CI
    NPD --> DI
    NPD -.->|"fallback"| NPC

    %% Styling
    classDef tierA fill:#4CAF50,color:white
    classDef tierB fill:#2196F3,color:white
    classDef tierC fill:#FF9800,color:white
    classDef tierD fill:#9E9E9E,color:white
    classDef memrl fill:#E1BEE7,color:black
    classDef terminal fill:#F44336,color:white

    class FD tierA
    class CP,CE,ILC,THR,AG,AC tierB
    class WG,WM,WS,WV,TR tierC
    class DC,DG tierD
    class EP,HG,FG,TPR memrl
    class END terminal
```

## Escalation Chains (Static)

```mermaid
flowchart LR
    subgraph Chain1["Worker Chain"]
        W1[worker_*] --> C1[coder_escalation] --> A1[architect_coding]
    end

    subgraph Chain2["Coder Chain"]
        C2[coder_escalation] --> A2[architect_coding]
    end

    subgraph Chain4["Ingest Chain"]
        I1[ingest_long_context] --> A3[architect_general]
    end

    subgraph Chain5["Frontdoor Chain"]
        F1[frontdoor] --> C3[coder_escalation] --> A4[architect_coding]
    end
```

## MemRL-Informed Dynamic Routing

```mermaid
flowchart TB
    subgraph Static["Static Edges (roles.py)"]
        direction LR
        S1["worker → coder"]
        S2["coder → architect"]
        S3["ingest → architect"]
    end

    subgraph Dynamic["Dynamic Weights (MemRL)"]
        direction TB
        Q["Q-value: 0.85<br/>'worker→architect'"]
        H["Hypothesis: 0.92<br/>'architect|code_refactor'"]
        F["Failure penalty: 0.3<br/>'coder:timeout'"]
    end

    subgraph Decision["Routing Decision"]
        D{{"confidence > 0.7?"}}
        D -->|"Yes"| LEARNED["Use learned target"]
        D -->|"No"| STATIC["Use static chain"]
    end

    Static --> Decision
    Dynamic --> Decision
```

## Pipeline Stages

```mermaid
flowchart LR
    subgraph Stages["Chat Pipeline Stages"]
        S1["1. Route"] --> S2["2. Preprocess"]
        S2 --> S3["3. Backend Init"]
        S3 --> S4["4. Mock?"]
        S4 -->|"no"| S5["5. Plan Review?"]
        S4 -->|"yes"| MOCK["Mock Response"]
        S5 --> S6["6. Vision?"]
        S6 --> S7["7. Delegate?"]
        S7 --> S8["8. REPL Loop"]
        S8 --> S9["9. Error Annotate"]
    end

    subgraph Escalation["Escalation Loop"]
        E1["Failure Detected"]
        E2["RoutingFacade.decide()"]
        E3{{"action?"}}
        E1 --> E2 --> E3
        E3 -->|"retry"| S8
        E3 -->|"escalate"| NEXT["Next Tier"]
        E3 -->|"fail"| FAIL["Terminal Failure"]
        NEXT --> S8
    end
```

## Source Files

| Component | File | Line |
|-----------|------|------|
| Role definitions | `src/roles.py` | 55-176 |
| Escalation map | `src/roles.py` | 274-292 |
| Tier map | `src/roles.py` | 251-271 |
| Escalation policy | `src/escalation.py` | 1-300 |
| Routing facade | `src/routing_facade.py` | 1-135 |
| Failure router | `src/failure_router.py` | 1-600 |
| Pipeline stages | `src/api/routes/chat_pipeline/` | — |
| MemRL retriever | `orchestration/repl_memory/retriever.py` | 1-300 |
| Hypothesis graph | `orchestration/repl_memory/hypothesis_graph.py` | 1-200 |
| Failure graph | `orchestration/repl_memory/failure_graph.py` | 1-200 |
