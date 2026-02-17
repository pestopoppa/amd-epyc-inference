"""
Hierarchical Local-Agent Orchestration.

A multi-tier agent system for CPU-optimized LLM inference on AMD EPYC hardware.

Packages:
    api: FastAPI application, routes, request/response models, structured logging
    backends: LLM inference backends (llama.cpp HTTP, completion, speculative)
    classifiers: Intent and complexity classification for task routing
    db: SQLite models for episodic memory, skills, and session storage
    graph: Pydantic-graph orchestration nodes (triage, plan, execute, review)
    llm_primitives: Low-level LLM call wrappers with retry and streaming
    metrics: Inference telemetry and performance counters
    models: Pydantic models for document processing pipeline
    pipeline_monitor: Real-time pipeline health and anomaly detection
    proactive_delegation: Complexity-driven task delegation to specialist agents
    prompt_builders: Role-specific prompt assembly (frontdoor, coder, architect)
    repl_environment: Sandboxed Python REPL with AST security and tool access
    services: Document processing, caching, corpus retrieval services
    session: Session persistence, checkpointing, and document caching
    tools: Callable tool implementations (code, data, file, web, canvas)
    vision: Image/video analysis with CLIP embeddings and batch processing
"""
