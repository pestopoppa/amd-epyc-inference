#!/usr/bin/env python3
"""LightOnOCR-2 document processing server."""

from __future__ import annotations

import argparse
import base64
import io
import logging
import re
import time

import pypdfium2 as pdfium
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("lightonocr-server")

# Lazy loading
processor = None
model = None
# Use HuggingFace ID (auto-download)
MODEL_ID = "lightonai/LightOnOCR-2-1B-bbox"


def get_model():
    """Lazy load the OCR model."""
    global processor, model
    if model is not None:
        return processor, model

    import torch
    from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor

    logger.info(f"Loading {MODEL_ID}...")
    start = time.time()

    # Determine device and dtype
    if torch.cuda.is_available():
        device = "cuda"
        dtype = torch.bfloat16
    else:
        device = "cpu"
        dtype = torch.float32  # BF16 on CPU may not work well

    logger.info(f"Using device: {device}, dtype: {dtype}")

    processor = LightOnOcrProcessor.from_pretrained(MODEL_ID)
    model = LightOnOcrForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
    ).to(device)

    logger.info(f"Model loaded in {time.time() - start:.2f}s")
    return processor, model


def render_pdf_page(pdf_bytes: bytes, page_num: int = 0, dpi: int = 200) -> Image.Image:
    """Render a PDF page to PIL Image at specified DPI."""
    pdf = pdfium.PdfDocument(pdf_bytes)
    page = pdf[page_num]
    scale = dpi / 72  # 72 is the default PDF DPI
    bitmap = page.render(scale=scale)
    return bitmap.to_pil()


def ocr_image(image: Image.Image, output_format: str = "bbox") -> dict:
    """Run OCR on a single image."""
    import torch

    proc, mdl = get_model()
    device = next(mdl.parameters()).device
    dtype = next(mdl.parameters()).dtype

    # Resize to model input size (1540px longest dimension)
    max_dim = 1540
    ratio = min(max_dim / image.width, max_dim / image.height)
    if ratio < 1:
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Use apply_chat_template with conversation format
    conversation = [{"role": "user", "content": [{"type": "image", "image": image}]}]
    inputs = proc.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Move to model device and dtype
    inputs = {
        k: v.to(device=device, dtype=dtype) if v.is_floating_point() else v.to(device)
        for k, v in inputs.items()
    }

    start = time.time()
    with torch.no_grad():
        output_ids = mdl.generate(**inputs, max_new_tokens=4096)

    # Extract only generated tokens (skip input)
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]
    text = proc.decode(generated_ids, skip_special_tokens=True)
    elapsed = time.time() - start

    # Parse bounding boxes from text (format: ![image](image_N.png) x1,y1,x2,y2)
    bboxes = []
    if output_format in ("bbox", "json"):
        for match in re.finditer(r"!\[image\]\(image_(\d+)\.png\)\s+(\d+),(\d+),(\d+),(\d+)", text):
            bboxes.append(
                {
                    "id": int(match.group(1)),
                    "x1": int(match.group(2)),
                    "y1": int(match.group(3)),
                    "x2": int(match.group(4)),
                    "y2": int(match.group(5)),
                    "normalized": True,  # coords are in [0, 1000] range
                }
            )

    return {"text": text, "bboxes": bboxes, "elapsed_sec": elapsed}


# FastAPI app
app = FastAPI(title="LightOnOCR-2 Server", version="1.0.0")


@app.on_event("startup")
async def startup():
    logger.info("Warming up model...")
    get_model()
    logger.info("Server ready")


@app.get("/health")
async def health():
    return {"status": "healthy", "model": MODEL_ID}


@app.post("/v1/document/ocr")
async def ocr_endpoint(
    image: str = Form(...),  # base64-encoded image
    output_format: str = Form(default="bbox"),
):
    """OCR a single image (base64-encoded)."""
    try:
        img_bytes = base64.b64decode(image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    result = ocr_image(img, output_format)
    return result


@app.post("/v1/document/pdf")
async def pdf_endpoint(
    file: UploadFile = File(...),
    output_format: str = Form(default="bbox"),
    max_pages: int = Form(default=100),
):
    """OCR an entire PDF document."""
    pdf_bytes = await file.read()

    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        num_pages = min(len(pdf), max_pages)
    except Exception as e:
        raise HTTPException(400, f"Invalid PDF: {e}")

    pages = []
    total_start = time.time()

    for i in range(num_pages):
        img = render_pdf_page(pdf_bytes, i)
        result = ocr_image(img, output_format)
        result["page"] = i + 1
        pages.append(result)

    total_elapsed = time.time() - total_start
    pages_per_sec = num_pages / total_elapsed if total_elapsed > 0 else 0

    return {
        "pages": pages,
        "total_pages": num_pages,
        "elapsed_sec": total_elapsed,
        "pages_per_sec": pages_per_sec,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9001)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
