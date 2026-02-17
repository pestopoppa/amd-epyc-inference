"""
Service modules for document processing, caching, and corpus retrieval.

This package provides backend services used by the orchestration pipeline:

Document Processing:
    - pdf_router: Intelligent routing between pdftotext and LightOnOCR
    - document_client: Async HTTP client for LightOnOCR server
    - document_chunker: Semantic chunking by markdown headers
    - document_preprocessor: Combined OCR + chunking + figure analysis
    - figure_analyzer: Vision-model figure analysis with bbox cropping
    - archive_extractor: Secure archive extraction with zip-bomb detection
    - lightonocr_server: LightOnOCR-2 PyTorch server
    - lightonocr_llama_server: LightOnOCR-2 GGUF server (llama-mtmd-cli)

Encoding & Compression:
    - toon_encoder: TOON encoding for 40-65% token reduction
    - prompt_compressor: LLMLingua-2 extractive token selection

Inference Support:
    - draft_cache: Content-addressed cache for two-stage summarization
    - corpus_retrieval: N-gram index retrieval for prompt-lookup acceleration
    - worker_pool: Heterogeneous worker pool for parallel task execution
"""
