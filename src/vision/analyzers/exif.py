"""EXIF metadata extraction analyzer."""

from __future__ import annotations

import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from src.vision.analyzers.base import Analyzer, AnalyzerResult
from src.vision.config import EXIFTOOL_TIMEOUT

logger = logging.getLogger(__name__)


class ExifAnalyzer(Analyzer):
    """Extract EXIF metadata from images.

    Uses PIL for basic EXIF and falls back to exiftool for comprehensive extraction.
    """

    def __init__(self, use_exiftool: bool = True, **config: Any):
        """Initialize EXIF analyzer.

        Args:
            use_exiftool: Whether to use exiftool for extraction (more complete).
            **config: Additional configuration.
        """
        super().__init__(**config)
        self.use_exiftool = use_exiftool

    @property
    def name(self) -> str:
        return "exif_extract"

    def initialize(self) -> None:
        """Check if exiftool is available."""
        if self.use_exiftool:
            try:
                result = subprocess.run(
                    ["exiftool", "-ver"],
                    capture_output=True,
                    text=True,
                    timeout=EXIFTOOL_TIMEOUT,
                )
                if result.returncode != 0:
                    logger.warning("exiftool not available, falling back to PIL")
                    self.use_exiftool = False
            except FileNotFoundError:
                logger.warning("exiftool not found, falling back to PIL")
                self.use_exiftool = False
            except subprocess.TimeoutExpired:
                self.use_exiftool = False

        super().initialize()

    def analyze(self, image: Image.Image, path: Path | None = None) -> AnalyzerResult:
        """Extract EXIF metadata from image.

        Args:
            image: PIL Image (used if path not available).
            path: Path to original file (preferred for complete EXIF).

        Returns:
            AnalyzerResult with EXIF data dict.
        """
        self.ensure_initialized()
        start = time.perf_counter()

        try:
            if self.use_exiftool and path and path.exists():
                exif_data = self._extract_with_exiftool(path)
            else:
                exif_data = self._extract_with_pil(image)

            elapsed = (time.perf_counter() - start) * 1000

            return AnalyzerResult(
                analyzer_name=self.name,
                success=True,
                data={"exif": exif_data},
                processing_time_ms=elapsed,
            )

        except Exception as e:
            logger.error(f"EXIF extraction failed: {e}")
            return AnalyzerResult(
                analyzer_name=self.name,
                success=False,
                error=str(e),
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

    def _extract_with_exiftool(self, path: Path) -> dict[str, Any]:
        """Extract EXIF using exiftool (more complete)."""
        import json

        result = subprocess.run(
            ["exiftool", "-j", "-n", str(path)],
            capture_output=True,
            text=True,
            timeout=EXIFTOOL_TIMEOUT,
        )

        if result.returncode != 0:
            raise RuntimeError(f"exiftool failed: {result.stderr}")

        data = json.loads(result.stdout)[0]

        # Normalize to our schema
        exif = {}

        # Date taken
        for key in ["DateTimeOriginal", "CreateDate", "ModifyDate"]:
            if key in data:
                try:
                    exif["taken_at"] = self._parse_exif_date(data[key])
                    break
                except ValueError:
                    pass

        # Camera info
        exif["camera_make"] = data.get("Make")
        exif["camera_model"] = data.get("Model")
        exif["focal_length"] = data.get("FocalLength")
        exif["aperture"] = data.get("FNumber")
        exif["iso"] = data.get("ISO")
        exif["orientation"] = data.get("Orientation")

        # Dimensions
        exif["width"] = data.get("ImageWidth")
        exif["height"] = data.get("ImageHeight")

        # GPS
        if "GPSLatitude" in data and "GPSLongitude" in data:
            exif["gps_lat"] = float(data["GPSLatitude"])
            exif["gps_lon"] = float(data["GPSLongitude"])

            # Handle hemisphere
            if data.get("GPSLatitudeRef") == "S":
                exif["gps_lat"] = -exif["gps_lat"]
            if data.get("GPSLongitudeRef") == "W":
                exif["gps_lon"] = -exif["gps_lon"]

        return {k: v for k, v in exif.items() if v is not None}

    def _extract_with_pil(self, image: Image.Image) -> dict[str, Any]:
        """Extract EXIF using PIL (basic)."""
        exif = {}

        # Get dimensions from image
        exif["width"] = image.width
        exif["height"] = image.height

        # Try to get EXIF data
        try:
            raw_exif = image._getexif()
            if raw_exif is None:
                return exif

            # Map tag IDs to names
            exif_data = {}
            for tag_id, value in raw_exif.items():
                tag = TAGS.get(tag_id, tag_id)
                exif_data[tag] = value

            # Date taken
            if "DateTimeOriginal" in exif_data:
                try:
                    exif["taken_at"] = self._parse_exif_date(exif_data["DateTimeOriginal"])
                except ValueError:
                    pass

            # Camera info
            exif["camera_make"] = exif_data.get("Make")
            exif["camera_model"] = exif_data.get("Model")
            exif["focal_length"] = str(exif_data.get("FocalLength", ""))
            exif["iso"] = exif_data.get("ISOSpeedRatings")
            exif["orientation"] = exif_data.get("Orientation")

            # F-number (aperture)
            if "FNumber" in exif_data:
                try:
                    exif["aperture"] = float(exif_data["FNumber"])
                except (TypeError, ValueError):
                    pass

            # GPS
            if "GPSInfo" in exif_data:
                gps = exif_data["GPSInfo"]
                gps_data = {}
                for tag_id, value in gps.items():
                    tag = GPSTAGS.get(tag_id, tag_id)
                    gps_data[tag] = value

                if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                    lat = self._convert_gps_coord(gps_data["GPSLatitude"])
                    lon = self._convert_gps_coord(gps_data["GPSLongitude"])

                    if gps_data.get("GPSLatitudeRef") == "S":
                        lat = -lat
                    if gps_data.get("GPSLongitudeRef") == "W":
                        lon = -lon

                    exif["gps_lat"] = lat
                    exif["gps_lon"] = lon

        except Exception as e:
            logger.debug(f"PIL EXIF extraction failed: {e}")

        return {k: v for k, v in exif.items() if v is not None}

    def _parse_exif_date(self, date_str: str) -> str:
        """Parse EXIF date format to ISO format."""
        # Common EXIF format: "2023:12:25 14:30:00"
        for fmt in [
            "%Y:%m:%d %H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y/%m/%d %H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(str(date_str), fmt)
                return dt.isoformat()
            except ValueError:
                continue
        raise ValueError(f"Could not parse date: {date_str}")

    def _convert_gps_coord(self, coord: tuple) -> float:
        """Convert GPS coordinate tuple to decimal degrees."""
        degrees = float(coord[0])
        minutes = float(coord[1])
        seconds = float(coord[2])
        return degrees + minutes / 60 + seconds / 3600
