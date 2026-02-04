"""Vision pipeline configuration and constants.

All vision-related paths are derived from ORCHESTRATOR_PATHS_VISION_DIR
(defaults to $LLM_ROOT/vision/) to comply with the project's strict
"no root filesystem writes" rule.

Values are sourced from the centralized config (src.config) with
module-level aliases for backward compatibility.
"""

from src.config import get_config

_cfg = get_config().vision
_tcfg = get_config().timeouts

# Base directory for all vision data
VISION_BASE_DIR = _cfg.base_dir

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
ARCFACE_MODEL_NAME = _cfg.arcface_model_name
CLIP_MODEL_NAME = _cfg.clip_model_name
SENTENCE_TRANSFORMER_MODEL = _cfg.sentence_transformer_model

# VL inference
LLAMA_MTMD_CLI = _cfg.llama_mtmd_cli
VL_MODEL_PATH = _cfg.vl_model_path
VL_MMPROJ_PATH = _cfg.vl_mmproj_path
# Vision server endpoints (preferred over CLI for production)
VL_SERVER_PORT = _cfg.vl_server_port
VL_ESCALATION_SERVER_PORT = _cfg.vl_escalation_server_port

# Processing limits
MAX_IMAGE_SIZE_MB = _cfg.max_image_size_mb
MAX_IMAGE_DIMENSION = _cfg.max_image_dimension
DEFAULT_BATCH_SIZE = _cfg.default_batch_size
MAX_CONCURRENT_WORKERS = _cfg.max_concurrent_workers
DEFAULT_VIDEO_FPS = _cfg.default_video_fps
DEFAULT_VL_MAX_TOKENS = _cfg.default_vl_max_tokens
DEFAULT_VL_THREADS = _cfg.default_vl_threads

# Timeout settings (seconds)
VL_INFERENCE_TIMEOUT = _tcfg.vision_inference
FFMPEG_VERSION_TIMEOUT = _tcfg.ffmpeg_version
FFMPEG_PROBE_TIMEOUT = _tcfg.ffmpeg_probe
FFMPEG_EXTRACT_TIMEOUT = _tcfg.ffmpeg_extract
EXIFTOOL_TIMEOUT = _tcfg.exiftool

# Thumbnail settings
THUMB_SIZE = _cfg.thumb_size
THUMB_QUALITY = _cfg.thumb_quality
TEMP_JPEG_QUALITY = _cfg.temp_jpeg_quality

# Face detection settings
FACE_MIN_CONFIDENCE = _cfg.face_min_confidence
FACE_EMBEDDING_DIM = _cfg.face_embedding_dim
FACE_IDENTIFICATION_THRESHOLD = _cfg.face_identification_threshold

# ONNX execution provider (CPU or CUDA)
ONNX_PROVIDERS = _cfg.onnx_providers

# Supported image extensions
SUPPORTED_IMAGE_EXTENSIONS = _cfg.supported_image_extensions

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
