"""
main.py — FastAPI application for TextSense OCR.

Endpoints:
  POST /analyze  — upload a file, run OCR, then use the LLM to extract
                   positive / negative points from the extracted text.
  GET  /         — serve the frontend (static/index.html)

Analysis backend:
  - Primary  : Parallel LLM execution via Gemini & Mistral (llm.py) + Validation layer
  - Fallback : HuggingFace sentiment pipeline (sentiment.py)
               activated automatically when the LLM call fails
"""

import logging
import os
import mimetypes
import asyncio
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Load .env before anything else so API keys are available
load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Configure logging before importing heavy modules
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline modules
# ---------------------------------------------------------------------------
from ocr import extract_text, extract_text_with_metadata   # noqa: E402
from llm import analyse_image_with_vision, analyse_with_models_parallel, summarise_batch  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_FILE_SIZE_MB = 20
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

ALLOWED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Log startup / shutdown."""
    gemini_status = "✓ set" if os.getenv("GEMINI_API_KEY") else "✗ missing"
    mistral_status = "✓ set" if os.getenv("MISTRAL_API_KEY") else "✗ missing"
    logger.info(f"TextSense OCR API starting up … (GEMINI_API_KEY: {gemini_status}, MISTRAL_API_KEY: {mistral_status})")
    yield
    logger.info("TextSense OCR API shutting down.")


app = FastAPI(
    title="TextSense OCR",
    description=(
        "Extract text from images/PDFs using PaddleOCR, then use an LLM "
        "to identify positive and negative points in the document."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# Serve the frontend at /static/...
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/static", StaticFiles(directory=_static_dir), name="static")

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _validate_upload(file: UploadFile, content: bytes) -> None:
    """
    Raise HTTPException if the file is too large or has an unsupported type.
    """
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum allowed size is {MAX_FILE_SIZE_MB} MB.",
        )


async def _analyze_one_upload(file: UploadFile) -> dict:
    """Analyze one uploaded file and return a normalized response payload."""
    content = await file.read()
    _validate_upload(file, content)
    filename = file.filename or "upload"
    ext = os.path.splitext(filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Accepted types: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            ),
        )

    # Fastest path for pictures: try vision model first.
    vision_already_failed = False
    first_vision_exc: Exception | None = None
    if ext in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        mime = file.content_type or mimetypes.guess_type(filename)[0] or "image/jpeg"
        try:
            result = await analyse_image_with_vision(content, mime)
            logger.info(f"Vision-first analysis complete for {filename}")
            return {"filename": filename, **result}
        except Exception as vision_exc:
            vision_already_failed = True
            first_vision_exc = vision_exc
            logger.warning(f"Vision-first path failed for {filename}: {vision_exc}. Falling back to local OCR.")

    # Local OCR path — use metadata version to get confidence + suspicious words
    ocr_succeeded = False
    raw_text = ""
    ocr_error = ""
    ocr_confidence = 1.0
    suspicious_words: list[str] = []
    try:
        ocr_meta = await asyncio.to_thread(extract_text_with_metadata, content, filename)
        raw_text       = ocr_meta["text"]
        ocr_confidence = ocr_meta.get("ocr_confidence", 1.0)
        suspicious_words = ocr_meta.get("suspicious_words", [])
        ocr_succeeded  = True
        logger.info(
            f"OCR complete for {filename} — "
            f"{len(raw_text)} chars, confidence={ocr_confidence:.0%}, "
            f"suspicious_words={len(suspicious_words)}"
        )
    except ValueError as exc:
        raise HTTPException(status_code=415, detail=str(exc))
    except RuntimeError as exc:
        ocr_succeeded = False
        raw_text = ""
        ocr_error = str(exc)
    except Exception as exc:
        logger.exception("Unexpected error during OCR")
        raise HTTPException(status_code=500, detail=f"OCR error: {str(exc)}")

    # Force vision fallback if OCR returned gibberish (e.g., handwriting)
    if ocr_succeeded and ocr_confidence < 0.40:
        logger.warning(f"OCR confidence too low ({ocr_confidence:.0%}). Forcing vision fallback (likely handwriting).")
        ocr_succeeded = False
        ocr_error = "OCR extracted text but confidence was too low (likely cursive handwriting)."

    if not ocr_succeeded:
        if ext == ".pdf":
            raise HTTPException(status_code=422, detail=ocr_error)

        # Only try vision again if vision-first did NOT already fail
        # (avoids burning quota on a second doomed API call)
        if not vision_already_failed:
            mime = file.content_type or mimetypes.guess_type(filename)[0] or "image/jpeg"
            try:
                result = await analyse_image_with_vision(content, mime)
                logger.info(f"Vision fallback analysis complete for {filename}")
                return {"filename": filename, **result}
            except Exception as vision_exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Local OCR failed: {ocr_error}. Vision fallback failed: {vision_exc}",
                )
        else:
            # Vision already failed earlier — report both errors, skip redundant call
            raise HTTPException(
                status_code=422,
                detail=f"Local OCR failed: {ocr_error}. Vision also failed earlier: {first_vision_exc}",
            )

    try:
        result = await analyse_with_models_parallel(
            raw_text,
            ocr_confidence=ocr_confidence,
            suspicious_words=suspicious_words,
        )
        logger.info(
            f"LLM analysis complete for {filename} — "
            f"{len(result['positive'])} positive, {len(result['negative'])} negative points"
        )
    except Exception as llm_exc:
        logger.warning(f"LLM analysis failed for {filename} ({llm_exc}); falling back to HuggingFace sentiment.")
        try:
            from sentiment import analyse_text
            hf_result = await asyncio.to_thread(analyse_text, raw_text)
            result = {**hf_result, "model": "HuggingFace (fallback)"}
        except Exception as hf_exc:
            logger.exception("Both LLM and HuggingFace analysis failed")
            raise HTTPException(
                status_code=500,
                detail=f"Analysis failed — LLM: {llm_exc} | HuggingFace: {hf_exc}",
            )


    return {"filename": filename, **result}

async def _analyze_one_upload_safe(file: UploadFile) -> dict:
    """Wrapper to catch exceptions and return an error dict instead of raising."""
    try:
        return await _analyze_one_upload(file)
    except HTTPException as exc:
        return {
            "_error": True,
            "filename": file.filename or "upload",
            "status": exc.status_code,
            "detail": str(exc.detail),
        }
    except Exception as exc:
        return {
            "_error": True,
            "filename": file.filename or "upload",
            "status": 500,
            "detail": str(exc),
        }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def index():
    """Serve the single-page frontend."""
    html_path = os.path.join(_static_dir, "index.html")
    if not os.path.isfile(html_path):
        raise HTTPException(status_code=404, detail="Frontend not found.")
    return FileResponse(html_path)


@app.post(
    "/analyze",
    summary="Extract text via OCR then analyse with LLM",
    response_description=(
        "JSON with raw_text, positive points, negative points, neutral sentences, "
        "and the model name used."
    ),
)
async def analyze(file: UploadFile = File(...)):
    """
    **POST /analyze**

    1. Uploads and validates the file (image or PDF, max 20 MB).
    2. Runs the OCR preprocessing pipeline to extract raw text.
    3. Sends the text to models (Mistral & Gemini) to extract points.
    4. Validates the result using Gemini as judge.

    Returns:
    ```json
    {
      "raw_text": "full extracted text",
      "positive": ["strength or compliment extracted by LLM", "..."],
      "negative": ["criticism or weakness extracted by LLM", "..."],
      "neutral":  [],
      "model": "Validated by gemini-1.5-pro (Sources: Mistral, Gemini)"
    }
    ```
    """
    result = await _analyze_one_upload(file)
    return JSONResponse(content=result)


@app.post(
    "/analyze-batch",
    summary="Analyze multiple files concurrently and return one combined summary",
)
async def analyze_batch(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    # Process all files concurrently
    tasks = [_analyze_one_upload_safe(file) for file in files]
    results = await asyncio.gather(*tasks)

    per_file: list[dict] = []
    errors: list[dict] = []

    for res in results:
        if res.get("_error"):
            errors.append(res)
        else:
            per_file.append(res)

    if not per_file:
        raise HTTPException(status_code=422, detail={"message": "All files failed.", "errors": errors})

    positive: list[str] = []
    negative: list[str] = []
    all_models: list[str] = []
    all_raw_text: list[str] = []
    for item in per_file:
        positive.extend(item.get("positive", []))
        negative.extend(item.get("negative", []))
        if item.get("model"):
            all_models.append(item["model"])
        if item.get("raw_text"):
            all_raw_text.append(f"### {item.get('filename', 'upload')}\n{item.get('raw_text', '')}")

    # De-duplicate while preserving order
    def _dedupe(values: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for v in values:
            key = str(v).strip().casefold()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(v).strip())
        return out

    positive = _dedupe(positive)
    negative = _dedupe(negative)
    neg_keys = {x.casefold() for x in negative}
    positive = [x for x in positive if x.casefold() not in neg_keys]

    summary = await summarise_batch(per_file, positive, negative)

    payload = {
        "summary": summary,
        "positive": positive,
        "negative": negative,
        "neutral": [],
        "raw_text": "\n\n".join(all_raw_text),
        "files": per_file,
        "errors": errors,
        "model": ", ".join(sorted(set(all_models))) if all_models else "unknown",
    }
    return JSONResponse(content=payload)


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4444, reload=True)
