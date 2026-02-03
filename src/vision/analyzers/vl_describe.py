"""Vision-Language description analyzer using llama-mtmd-cli."""

from __future__ import annotations

import logging
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from PIL import Image

from src.vision.analyzers.base import Analyzer, AnalyzerResult
from src.vision.config import (
    LLAMA_MTMD_CLI,
    VL_MODEL_PATH,
    VL_MMPROJ_PATH,
    DEFAULT_VL_MAX_TOKENS,
    DEFAULT_VL_THREADS,
    VL_INFERENCE_TIMEOUT,
    TEMP_JPEG_QUALITY,
    VISION_CACHE_DIR,
)

logger = logging.getLogger(__name__)


class VLDescribeAnalyzer(Analyzer):
    """Generate natural language descriptions of images using VL model.

    Uses llama-mtmd-cli for inference with Qwen2.5-VL-7B by default.
    Can be configured to use different models/prompts.
    """

    def __init__(
        self,
        prompt: str = "Describe this image briefly. Note people, setting, and activities.",
        max_tokens: int = DEFAULT_VL_MAX_TOKENS,
        model_path: Path | str | None = None,
        mmproj_path: Path | str | None = None,
        threads: int = DEFAULT_VL_THREADS,
        **config: Any,
    ):
        """Initialize VL description analyzer.

        Args:
            prompt: Prompt for the VL model.
            max_tokens: Maximum tokens to generate.
            model_path: Path to GGUF model (default: Qwen2.5-VL-7B).
            mmproj_path: Path to multimodal projector.
            threads: Number of threads for inference.
            **config: Additional configuration.
        """
        super().__init__(**config)
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.model_path = Path(model_path) if model_path else VL_MODEL_PATH
        self.mmproj_path = Path(mmproj_path) if mmproj_path else VL_MMPROJ_PATH
        self.threads = threads

    @property
    def name(self) -> str:
        return "vl_describe"

    def initialize(self) -> None:
        """Verify CLI and model files exist."""
        if not LLAMA_MTMD_CLI.exists():
            raise FileNotFoundError(f"llama-mtmd-cli not found at {LLAMA_MTMD_CLI}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"VL model not found at {self.model_path}")
        if not self.mmproj_path.exists():
            raise FileNotFoundError(f"mmproj not found at {self.mmproj_path}")

        super().initialize()

    def analyze(self, image: Image.Image, path: Path | None = None) -> AnalyzerResult:
        """Generate description for image.

        Args:
            image: PIL Image to describe.
            path: Optional path to original file (used directly if available).

        Returns:
            AnalyzerResult with description text.
        """
        self.ensure_initialized()
        start = time.perf_counter()

        try:
            # Use original path if available, otherwise save temp file
            if path and path.exists():
                image_path = str(path)
                temp_file = None
            else:
                VISION_CACHE_DIR.mkdir(parents=True, exist_ok=True)
                temp_file = tempfile.NamedTemporaryFile(
                    suffix=".jpg",
                    dir=VISION_CACHE_DIR,
                    delete=False,
                )
                image.save(temp_file.name, format="JPEG", quality=TEMP_JPEG_QUALITY)
                image_path = temp_file.name

            # Build command
            cmd = [
                str(LLAMA_MTMD_CLI),
                "-m",
                str(self.model_path),
                "--mmproj",
                str(self.mmproj_path),
                "--image",
                image_path,
                "-p",
                self.prompt,
                "-n",
                str(self.max_tokens),
                "-t",
                str(self.threads),
                "--temp",
                "0.0",
                "--no-display-prompt",
            ]

            # Run inference
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=VL_INFERENCE_TIMEOUT,
            )

            # Clean up temp file
            if temp_file:
                Path(temp_file.name).unlink(missing_ok=True)

            if result.returncode != 0:
                error = result.stderr or "Unknown error"
                logger.error(f"VL inference failed: {error}")
                return AnalyzerResult(
                    analyzer_name=self.name,
                    success=False,
                    error=error,
                    processing_time_ms=(time.perf_counter() - start) * 1000,
                )

            # Parse output (strip prompt echo if present)
            description = result.stdout.strip()

            # Remove common artifacts
            description = self._clean_output(description)

            elapsed = (time.perf_counter() - start) * 1000

            return AnalyzerResult(
                analyzer_name=self.name,
                success=True,
                data={"description": description},
                processing_time_ms=elapsed,
            )

        except subprocess.TimeoutExpired:
            return AnalyzerResult(
                analyzer_name=self.name,
                success=False,
                error="VL inference timed out",
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"VL description failed: {e}")
            return AnalyzerResult(
                analyzer_name=self.name,
                success=False,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

    def _clean_output(self, text: str) -> str:
        """Clean VL model output artifacts.

        Args:
            text: Raw model output.

        Returns:
            Cleaned description text.
        """
        # Remove thinking tags if present
        import re

        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

        # Remove assistant/user prefixes
        text = re.sub(r"^(Assistant:|assistant:|User:|user:)\s*", "", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text


class VLOCRAnalyzer(VLDescribeAnalyzer):
    """Extract text from images using VL model.

    Specialized prompt for OCR-like text extraction.
    """

    def __init__(self, **config: Any):
        prompt = (
            "Read and transcribe all visible text in this image exactly as it appears. "
            "Preserve formatting including line breaks and indentation where possible. "
            "If the image contains mathematical formulas, transcribe them in LaTeX format."
        )
        super().__init__(prompt=prompt, max_tokens=1024, **config)

    @property
    def name(self) -> str:
        return "vl_ocr"


class VLStructuredAnalyzer(VLDescribeAnalyzer):
    """Extract structured data from images (forms, receipts, tables).

    Returns JSON-formatted extraction results.
    """

    def __init__(self, schema_hint: str = "", **config: Any):
        """Initialize structured extraction analyzer.

        Args:
            schema_hint: Optional JSON schema or field hints for extraction.
            **config: Additional configuration.
        """
        self.schema_hint = schema_hint
        prompt = (
            "Extract structured information from this image and return it as valid JSON. "
            "Include fields like: date, total, items, names, addresses, etc. as applicable. "
            f"{schema_hint}"
        )
        super().__init__(prompt=prompt, max_tokens=1024, **config)

    @property
    def name(self) -> str:
        return "vl_structured"

    def analyze(self, image: Image.Image, path: Path | None = None) -> AnalyzerResult:
        """Extract structured data from image."""
        result = super().analyze(image, path)

        if result.success and result.data.get("description"):
            # Try to parse as JSON
            import json

            try:
                # Find JSON in the response
                text = result.data["description"]
                # Look for JSON block
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]

                structured = json.loads(text.strip())
                result.data["structured"] = structured
            except json.JSONDecodeError:
                # Return raw text if JSON parsing fails
                result.data["structured"] = None
                result.data["parse_error"] = "Failed to parse JSON from response"

        return result
