"""Vision pipeline configuration and constants.

All vision-related paths point to /mnt/raid0/llm/vision/ to comply with
the project's strict "no root filesystem writes" rule.
"""

from pathlib import Path

# Base directory for all vision data
VISION_BASE_DIR = Path("/mnt/raid0/llm/vision")

# Subdirectories
VISION_DATA_DIR = VISION_BASE_DIR / "data"
VISION_THUMBS_DIR = VISION_BASE_DIR / "thumbs"
VISION_MODELS_DIR = VISION_BASE_DIR / "models"
VISION_CACHE_DIR = VISION_BASE_DIR / "cache"
VISION_LOGS_DIR = VISION_BASE_DIR / "logs"
VISION_TEST_IMAGES_DIR = VISION_BASE_DIR / "test_images"

# Database paths
CHROMA_PATH = VISION_DATA_DIR / "chroma"
SQLITE_PATH = VISION_DATA_DIR / "sqlite" / "vision.db"

# Model paths
ARCFACE_MODEL_NAME = "buffalo_l"
CLIP_MODEL_NAME = "ViT-B/32"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# VL inference
LLAMA_MTMD_CLI = Path("/mnt/raid0/llm/llama.cpp/build/bin/llama-mtmd-cli")
VL_MODEL_PATH = Path("/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf")
VL_MMPROJ_PATH = Path("/mnt/raid0/llm/lmstudio/models/lmstudio-community/Qwen2.5-VL-7B-Instruct-GGUF/mmproj-model-f16.gguf")
# Vision server endpoints (preferred over CLI for production)
VL_SERVER_PORT = 8086
VL_ESCALATION_SERVER_PORT = 8087

# Processing limits
MAX_IMAGE_SIZE_MB = 20
MAX_IMAGE_DIMENSION = 4096
DEFAULT_BATCH_SIZE = 100
MAX_CONCURRENT_WORKERS = 4
DEFAULT_VIDEO_FPS = 1.0
DEFAULT_VL_MAX_TOKENS = 512
DEFAULT_VL_THREADS = 8

# Timeout settings (seconds)
VL_INFERENCE_TIMEOUT = 120
FFMPEG_VERSION_TIMEOUT = 5
FFMPEG_PROBE_TIMEOUT = 30
FFMPEG_EXTRACT_TIMEOUT = 600
EXIFTOOL_TIMEOUT = 30

# Thumbnail settings
THUMB_SIZE = (256, 256)
THUMB_QUALITY = 85
TEMP_JPEG_QUALITY = 95

# Face detection settings
FACE_MIN_CONFIDENCE = 0.9
FACE_EMBEDDING_DIM = 512  # ArcFace
FACE_IDENTIFICATION_THRESHOLD = 0.6

# ONNX execution provider (CPU or CUDA)
ONNX_PROVIDERS = ["CPUExecutionProvider"]

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = ["jpg", "jpeg", "png", "heic", "webp", "bmp", "tiff"]

# ChromaDB collection names
COLLECTION_FACES = "faces"
COLLECTION_DESCRIPTIONS = "descriptions"
COLLECTION_IMAGES = "images"


def ensure_directories() -> None:
    """Create all required directories if they don't exist."""
    dirs = [
        VISION_DATA_DIR,
        VISION_THUMBS_DIR,
        VISION_MODELS_DIR,
        VISION_CACHE_DIR,
        VISION_LOGS_DIR,
        VISION_TEST_IMAGES_DIR,
        CHROMA_PATH,
        SQLITE_PATH.parent,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
