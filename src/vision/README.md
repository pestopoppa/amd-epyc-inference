# Vision Processing Pipeline

Modular vision processing infrastructure for image analysis, face recognition, and semantic search.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    VISION PROCESSING PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   INPUT     │    │  ANALYZERS  │    │   OUTPUT    │         │
│  │  ADAPTERS   │    │  (Plugins)  │    │   SINKS     │         │
│  ├─────────────┤    ├─────────────┤    ├─────────────┤         │
│  │ • Single    │    │ • Face Det  │    │ • JSON      │         │
│  │ • Batch     │───▶│ • Face Emb  │───▶│ • SQLite    │         │
│  │ • Video     │    │ • VL Desc   │    │ • ChromaDB  │         │
│  │ • Base64    │    │ • EXIF      │    │ • Thumbnail │         │
│  └─────────────┘    │ • CLIP      │    └─────────────┘         │
│                     └─────────────┘                             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    SEARCH / RETRIEVAL                        ││
│  ├─────────────────────────────────────────────────────────────┤│
│  │ • Text search (description matching via embeddings)          ││
│  │ • Face search (ArcFace embedding similarity)                 ││
│  │ • Visual similarity (CLIP embeddings)                        ││
│  │ • Metadata filters (date, location, person)                  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Directory Structure

```
src/vision/
├── README.md              # This file
├── __init__.py            # Module exports
├── config.py              # Paths, constants, timeouts
├── models.py              # Pydantic request/response models
├── pipeline.py            # Main pipeline orchestration
├── batch.py               # Batch job management
├── search.py              # Unified search interface
├── video.py               # Video frame extraction
├── clustering.py          # HDBSCAN face clustering service
│
└── analyzers/             # Modular analyzer plugins
    ├── __init__.py        # Analyzer registry
    ├── base.py            # Abstract base class
    ├── insightface_loader.py  # Shared model singleton
    ├── face_detect.py     # Face detection (InsightFace)
    ├── face_embed.py      # Face embeddings (ArcFace 512-dim)
    ├── vl_describe.py     # VL descriptions (llama-mtmd-cli)
    ├── exif.py            # EXIF metadata extraction
    └── clip_embed.py      # CLIP visual embeddings
```

## Quick Start

```python
from src.vision.pipeline import get_pipeline
from src.vision.models import AnalyzerType

# Initialize pipeline
pipeline = get_pipeline()
pipeline.initialize([AnalyzerType.FACE_DETECT, AnalyzerType.VL_DESCRIBE])

# Analyze single image
result = pipeline.analyze("/path/to/image.jpg")

print(f"Description: {result.description}")
print(f"Faces found: {len(result.faces)}")
```

## Analyzers

| Analyzer | Output | Model |
|----------|--------|-------|
| `face_detect` | Bounding boxes, confidence | InsightFace/RetinaFace |
| `face_embed` | 512-dim ArcFace vectors | InsightFace/buffalo_l |
| `vl_describe` | Natural language description | Qwen2.5-VL-7B (via llama-mtmd-cli) |
| `vl_ocr` | Extracted text | Qwen2.5-VL-7B |
| `vl_structured` | JSON extraction | Qwen2.5-VL-7B |
| `exif_extract` | Metadata (date, GPS, camera) | exiftool / PIL |
| `clip_embed` | 512-dim visual embeddings | CLIP ViT-B/32 |

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/v1/vision/analyze` | POST | Single image analysis |
| `/v1/vision/batch` | POST | Start batch job |
| `/v1/vision/batch/{job_id}` | GET | Check job status |
| `/v1/vision/search` | POST | Search indexed content |
| `/v1/vision/faces` | GET | List known persons |
| `/v1/vision/faces/{id}` | PUT | Update person (name, merge) |
| `/v1/vision/faces/cluster` | POST | Cluster unlabeled faces |
| `/v1/vision/video/analyze` | POST | Video frame analysis |
| `/v1/vision/stats` | GET | Pipeline statistics |

## Storage

### SQLite (`/mnt/raid0/llm/vision/data/sqlite/vision.db`)

| Table | Purpose |
|-------|---------|
| `photos` | Image metadata (path, hash, dimensions, EXIF, description) |
| `faces` | Detected faces with bounding boxes |
| `persons` | Named individuals for face identification |
| `videos` | Indexed videos with duration |
| `video_frames` | Extracted frames with timestamps |

### ChromaDB (`/mnt/raid0/llm/vision/data/chroma/`)

| Collection | Embedding Dim | Purpose |
|------------|---------------|---------|
| `faces` | 512 (ArcFace) | Face similarity search |
| `descriptions` | 384 (MiniLM) | Text search on descriptions |
| `images` | 512 (CLIP) | Visual similarity search |

## Configuration

Key constants in `config.py`:

```python
# Paths (all under /mnt/raid0/llm/vision/)
VISION_DATA_DIR      # SQLite, ChromaDB storage
VISION_THUMBS_DIR    # Generated thumbnails
VISION_MODELS_DIR    # InsightFace models
VISION_CACHE_DIR     # Temp files, embeddings cache

# Timeouts (seconds)
VL_INFERENCE_TIMEOUT = 120      # VL model inference
FFMPEG_EXTRACT_TIMEOUT = 600    # Video frame extraction
EXIFTOOL_TIMEOUT = 30           # EXIF extraction

# Processing
MAX_IMAGE_DIMENSION = 4096      # Auto-resize larger images
FACE_MIN_CONFIDENCE = 0.9       # Detection threshold
THUMB_SIZE = (256, 256)         # Thumbnail dimensions
```

## Design Patterns

### Shared Model Loader

Face analyzers share a singleton InsightFace instance to avoid duplicate 500MB model loads:

```python
# analyzers/insightface_loader.py
from src.vision.analyzers.insightface_loader import get_face_app

# Both face_detect and face_embed use the same instance
app = get_face_app()  # Lazy-loaded singleton
```

### Session Context Manager

Database operations use the managed session context:

```python
from src.db.models.vision import managed_session

with managed_session() as session:
    session.add(photo)
    # Auto-commits on success, rolls back on exception
```

### Analyzer Base Class

All analyzers extend the base class with standardized result handling:

```python
from src.vision.analyzers.base import Analyzer

class MyAnalyzer(Analyzer):
    @property
    def name(self) -> str:
        return "my_analyzer"

    def analyze(self, image: Image.Image, path: Path | None = None) -> AnalyzerResult:
        start = time.perf_counter()
        try:
            data = self._do_analysis(image)
            return self._success_result(data, start)
        except Exception as e:
            return self._error_result(e, start)
```

## Dependencies

### System

- ffmpeg (video processing)
- exiftool (EXIF extraction)
- tesseract (backup OCR)

### Python

- insightface (face detection/embedding)
- chromadb (vector storage)
- sentence-transformers (text embeddings)
- hdbscan (face clustering)
- pillow (image processing)

### Models

- InsightFace buffalo_l (~340MB)
- all-MiniLM-L6-v2 (~80MB)
- Qwen2.5-VL-7B GGUF (~4.4GB, external)

## Testing

```bash
# Run vision tests
pytest tests/vision/ -v

# Run with coverage
pytest tests/vision/ --cov=src/vision --cov-report=term-missing
```

## Related Documentation

- [Handoff Document](../../handoffs/active/vision-pipeline.md)
- [Progress Log](../../progress/2026-01/2026-01-16.md)
