"""Document processing tool methods for the REPL environment.

Provides mixin with: ocr_document, analyze_figure, extract_figure.
"""

from __future__ import annotations

from typing import Any


class _DocumentToolsMixin:
    """Mixin providing document processing tools (_ocr_document, _analyze_figure, _extract_figure).

    Required attributes (provided by REPLEnvironment.__init__):
        config: REPLConfig — environment configuration
        _exploration_calls: int — exploration call counter
        _exploration_log: ExplorationLog — exploration event history
        progress_logger: ProgressLogger | None — progress tracking service
        task_id: str — task identifier for logging
        _validate_file_path: Callable[[str], tuple[bool, str | None]] — path validation method
    """

    def _ocr_document(self, path: str) -> str:
        """Extract text and figure bounding boxes from a PDF document.

        Uses the LightOnOCR-2 server for fast, accurate OCR processing.

        Args:
            path: Absolute path to the PDF file.

        Returns:
            JSON string with extracted text and figure locations.
        """
        self._exploration_calls += 1
        import json
        import requests

        # Validate path
        is_valid, error = self._validate_file_path(path)
        if not is_valid:
            return f"[ERROR: {error}]"

        if not path.lower().endswith(".pdf"):
            return "[ERROR: Only PDF files are supported. Use peek() for text files.]"

        try:
            with open(path, "rb") as f:
                files = {"file": (path.split("/")[-1], f, "application/pdf")}
                resp = requests.post(
                    "http://localhost:9001/v1/document/pdf",
                    files=files,
                    data={"max_pages": 100, "dpi": 200},
                    timeout=600,  # 10 min timeout for large documents
                )

            if resp.status_code != 200:
                return f"[ERROR: OCR server returned {resp.status_code}: {resp.text[:200]}]"

            data = resp.json()

            # Combine text from all pages
            full_text = ""
            all_figures = []

            for page_data in data.get("pages", []):
                page_num = page_data.get("page", 0)
                page_text = page_data.get("text", "")
                full_text += f"\n--- Page {page_num} ---\n{page_text}"

                # Collect figure bounding boxes
                for bbox in page_data.get("bboxes", []):
                    all_figures.append({
                        "page": page_num,
                        "id": bbox.get("id"),
                        "bbox": [bbox.get("x1"), bbox.get("y1"),
                                 bbox.get("x2"), bbox.get("y2")],
                    })

            result = {
                "full_text": full_text.strip()[:50000],  # Cap at 50K chars
                "total_pages": data.get("total_pages", 0),
                "figures": all_figures,
                "elapsed_sec": data.get("elapsed_sec", 0),
            }

            self._exploration_log.add_event("ocr_document", {"path": path}, result)

            # Log formalizer usage to progress logger
            if self.progress_logger is not None:
                self.progress_logger.log_formalizer_invocation(
                    task_id=self.task_id,
                    service="document_formalizer",
                    endpoint="/v1/document/pdf",
                    pages=data.get("total_pages", 0),
                    elapsed_sec=data.get("elapsed_sec", 0),
                    success=True,
                )

            return json.dumps(result, indent=2)

        except requests.exceptions.ConnectionError:
            if self.progress_logger is not None:
                self.progress_logger.log_formalizer_invocation(
                    task_id=self.task_id,
                    service="document_formalizer",
                    endpoint="/v1/document/pdf",
                    success=False,
                    error="server not running",
                )
            return "[ERROR: LightOnOCR server not running on port 9001. Start with: python src/services/lightonocr_llama_server.py]"
        except FileNotFoundError:
            return f"[ERROR: File not found: {path}]"
        except Exception as e:
            if self.progress_logger is not None:
                self.progress_logger.log_formalizer_invocation(
                    task_id=self.task_id,
                    service="document_formalizer",
                    endpoint="/v1/document/pdf",
                    success=False,
                    error=str(e),
                )
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _analyze_figure(self, image_path: str, prompt: str = "Describe this figure in detail") -> str:
        """Analyze an image or extracted figure with the vision model.

        Args:
            image_path: Path to the image file (PNG, JPG, etc.)
            prompt: Analysis prompt/question about the image.

        Returns:
            Description of the image content.
        """
        self._exploration_calls += 1
        import json
        import requests

        # Validate path
        is_valid, error = self._validate_file_path(image_path)
        if not is_valid:
            return f"[ERROR: {error}]"

        try:
            resp = requests.post(
                "http://localhost:8000/v1/vision/analyze",
                json={
                    "image_path": image_path,
                    "vl_prompt": prompt,
                    "analyzers": ["vl_describe"],
                },
                timeout=120,
            )

            if resp.status_code != 200:
                return f"[ERROR: Vision API returned {resp.status_code}: {resp.text[:200]}]"

            data = resp.json()
            description = data.get("vl_description", data.get("description", ""))

            self._exploration_log.add_event(
                "analyze_figure",
                {"image_path": image_path, "prompt": prompt},
                description
            )
            return description

        except requests.exceptions.ConnectionError:
            return "[ERROR: Vision API not available on localhost:8000]"
        except FileNotFoundError:
            return f"[ERROR: File not found: {image_path}]"
        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"

    def _extract_figure(
        self,
        pdf_path: str,
        page: int,
        bbox: list[int],
        output_path: str | None = None,
    ) -> str:
        """Extract a figure from a PDF given its bounding box.

        Args:
            pdf_path: Path to the PDF file.
            page: Page number (1-indexed).
            bbox: Bounding box as [x1, y1, x2, y2] in 0-1000 range.
            output_path: Optional output path. If None, saves to /tmp.

        Returns:
            Path to the extracted figure image.
        """
        self._exploration_calls += 1
        import os
        import tempfile

        # Validate paths
        is_valid, error = self._validate_file_path(pdf_path)
        if not is_valid:
            return f"[ERROR: {error}]"

        if output_path:
            is_valid, error = self._validate_file_path(output_path)
            if not is_valid:
                return f"[ERROR: {error}]"

        try:
            import pypdfium2 as pdfium
            from PIL import Image
        except ImportError:
            return "[ERROR: pypdfium2 or Pillow not installed]"

        try:
            pdf = pdfium.PdfDocument(pdf_path)
            if page < 1 or page > len(pdf):
                return f"[ERROR: Page {page} out of range (1-{len(pdf)})]"

            # Render page at high DPI
            pdf_page = pdf[page - 1]
            scale = 300 / 72  # 300 DPI
            bitmap = pdf_page.render(scale=scale)
            img = bitmap.to_pil()

            # Convert normalized coords (0-1000) to pixel coords
            width, height = img.size
            x1 = int(bbox[0] * width / 1000)
            y1 = int(bbox[1] * height / 1000)
            x2 = int(bbox[2] * width / 1000)
            y2 = int(bbox[3] * height / 1000)

            # Crop the figure
            cropped = img.crop((x1, y1, x2, y2))

            # Save to output path or temp
            if output_path:
                save_path = output_path
            else:
                from src.config import get_config
                tmp_dir = str(get_config().paths.tmp_dir)
                fd, save_path = tempfile.mkstemp(suffix=".png", dir=tmp_dir)
                os.close(fd)

            cropped.save(save_path, "PNG")

            self._exploration_log.add_event(
                "extract_figure",
                {"pdf_path": pdf_path, "page": page, "bbox": bbox},
                save_path
            )
            return save_path

        except Exception as e:
            return f"[ERROR: {type(e).__name__}: {e}]"
