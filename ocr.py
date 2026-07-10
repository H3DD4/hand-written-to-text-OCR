"""
ocr.py — OCR extraction and image preprocessing pipeline for TextSense OCR.

Preprocessing pipeline (applied before OCR):
  1. Grayscale conversion
  2. Deskew (Hough-line rotation correction)
  3. Denoise (fastNlMeansDenoising or median blur fallback)
  4. Smart binarization (Otsu for high-contrast, adaptive for low-contrast)
  5. Upscale if width < 1000 px
  6. CLAHE contrast enhancement

OCR engines:
  - PaddleOCR 3.x  (primary, multilingual)
  - pytesseract     (fallback, eng+fra+ara)

PDF handling:
  - PyMuPDF extracts native text layer when present (no OCR needed)
  - pdf2image converts scanned PDFs to images for OCR
"""

import io
import re
import math
import logging
import numpy as np
from pathlib import Path

import cv2
from PIL import Image

logger = logging.getLogger(__name__)

# ── PaddleOCR — primary engine ──────────────────────────────────────────────
PADDLE_AVAILABLE = False
_paddle = None

try:
    from paddleocr import PaddleOCR as _PaddleOCRCls

    def _init_paddle():
        """Initialise PaddleOCR lazily (first call downloads models ~30 MB)."""
        global _paddle, PADDLE_AVAILABLE
        if _paddle is None:
            try:
                # PaddleOCR API changed between 2.x and 3.x.
                # Try 3.x-style args first, then fall back to 2.x-style args.
                try:
                    _paddle = _PaddleOCRCls(
                        lang="en",
                        use_textline_orientation=True,
                    )
                except TypeError:
                    _paddle = _PaddleOCRCls(
                        use_angle_cls=True,
                        lang="en",
                        use_gpu=False,
                    )
                PADDLE_AVAILABLE = True
                logger.info("PaddleOCR initialised successfully.")
            except Exception as exc:
                logger.warning(f"PaddleOCR init failed: {exc}")
                PADDLE_AVAILABLE = False

except ImportError as exc:
    logger.warning(f"PaddleOCR not importable: {exc}")
    def _init_paddle():
        pass

# ── pytesseract — fallback OCR engine ───────────────────────────────────────
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    pytesseract.get_tesseract_version()   # will raise if binary not on PATH
    TESSERACT_AVAILABLE = True
    logger.info("Tesseract available as fallback OCR engine.")
except Exception as exc:
    logger.warning(f"Tesseract not available: {exc}")

# ── PyMuPDF — native PDF text extraction ────────────────────────────────────
FITZ_AVAILABLE = False
try:
    import fitz  # PyMuPDF
    FITZ_AVAILABLE = True
except ImportError:
    pass

# ── pdf2image — scanned PDF → PIL images ────────────────────────────────────
PDF2IMAGE_AVAILABLE = False
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _deskew(img: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(img, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, math.pi / 180,
                             threshold=80, minLineLength=80, maxLineGap=10)
    if lines is None:
        return img

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        dx = x2 - x1
        if dx == 0:
            continue
        angle = math.degrees(math.atan2(y2 - y1, dx))
        if abs(angle) < 45:
            angles.append(angle)

    if not angles:
        return img

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img

    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _denoise(img: np.ndarray) -> np.ndarray:
    try:
        return cv2.fastNlMeansDenoising(img, h=10,
                                         templateWindowSize=7,
                                         searchWindowSize=21)
    except cv2.error:
        return cv2.medianBlur(img, 3)


def _binarize(img: np.ndarray) -> np.ndarray:
    std = float(img.std())
    if std > 40:
        _, binary = cv2.threshold(img, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        binary = cv2.adaptiveThreshold(img, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY,
                                        blockSize=15, C=8)
    return binary


def _upscale_if_needed(img: np.ndarray, min_width: int = 1000) -> np.ndarray:
    h, w = img.shape[:2]
    if w < min_width:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)
    return img


def _clahe(img: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img)


def preprocess_image(pil_image: Image.Image) -> np.ndarray:
    """Pass 1 — Full preprocessing pipeline. Returns preprocessed grayscale ndarray."""
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = _to_grayscale(img)
    img = _deskew(img)
    img = _denoise(img)
    img = _binarize(img)
    img = _upscale_if_needed(img)
    img = _clahe(img)
    return img


def _preprocess_pass2(pil_image: Image.Image) -> np.ndarray:
    """
    Pass 2 — Soft enhance, no binarization.
    Works better for colour-gradient documents where binarization loses contrast cues.
    """
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = _to_grayscale(img)
    img = _deskew(img)
    img = _upscale_if_needed(img)
    img = _clahe(img)
    return img


def _preprocess_pass3(pil_image: Image.Image) -> np.ndarray:
    """
    Pass 3 — Inverted binarization.
    Works better for dark-background / white-text or watermarked documents.
    """
    img = np.array(pil_image.convert("RGB"))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = _to_grayscale(img)
    img = _deskew(img)
    img = _denoise(img)
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img = _upscale_if_needed(binary)
    img = _clahe(img)
    return img


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_NOISE_RE = re.compile(
    r"[^\w\s"
    r"\u0600-\u06FF"         # Arabic
    r"\u0750-\u077F"         # Arabic supplement
    r"\u00C0-\u024F"         # Latin Extended A/B
    r".,;:!?'\"\-_()\[\]{}/<>@#$%&*+=|\\~`^°±×÷«»…–—"
    r"]",
    re.UNICODE,
)


def _clean_text(raw: str) -> str:
    cleaned = _NOISE_RE.sub(" ", raw)
    lines = [re.sub(r"[ \t]+", " ", ln).strip()
             for ln in cleaned.splitlines()]
    result: list[str] = []
    prev_blank = False
    for line in lines:
        if not line:
            if not prev_blank:
                result.append("")
            prev_blank = True
        else:
            result.append(line)
            prev_blank = False
    return "\n".join(result).strip()


# ---------------------------------------------------------------------------
# PaddleOCR result parser  (handles 2.x and 3.x result shapes)
# ---------------------------------------------------------------------------

def _parse_paddle_result(result) -> list[str]:
    """
    Extract text strings from whatever shape PaddleOCR returns.

    PaddleOCR 2.x: result = [[bbox, (text, conf)], ...]
    PaddleOCR 3.x: result = [[[bbox, (text, conf)], ...]]   (wrapped in list)
                   or: list of OCRResult objects
    """
    lines: list[str] = []

    if result is None:
        return lines

    # Unwrap outer list if needed (PaddleOCR 3.x wraps per-page)
    page = result
    if isinstance(result, list) and len(result) > 0:
        first = result[0]
        # If the first element is also a list and its first item looks like a
        # page (list of [bbox, (text, score)] entries), unwrap one level.
        if isinstance(first, list) and len(first) > 0 and isinstance(first[0], list):
            page = first

    if page is None:
        return lines

    for item in page:
        if item is None:
            continue
        try:
            # Standard format: [bbox, (text, score)]
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                text_info = item[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text, conf = text_info[0], text_info[1]
                elif isinstance(text_info, str):
                    text, conf = text_info, 1.0
                else:
                    continue
                text = str(text).strip()
                conf = float(conf)
                if conf >= 0.45 and text:
                    lines.append(text)
            # Object-based format (some PaddleOCR 3.x versions)
            elif hasattr(item, "rec_text") and hasattr(item, "rec_score"):
                if float(item.rec_score) >= 0.45 and item.rec_text.strip():
                    lines.append(item.rec_text.strip())
        except Exception as exc:
            logger.debug(f"Skipping OCR block (parse error): {exc}")

    return lines


# ---------------------------------------------------------------------------
# OCR engines
# ---------------------------------------------------------------------------

def _run_paddleocr(img: np.ndarray) -> str:
    """Run PaddleOCR on a grayscale image and return concatenated text."""
    global _paddle, PADDLE_AVAILABLE

    _init_paddle()
    if not PADDLE_AVAILABLE or _paddle is None:
        return ""

    try:
        # PaddleOCR 3.x accepts kwargs only; 2.x supports cls flag.
        # Try cls first for backward compatibility, then plain call.
        try:
            result = _paddle.ocr(img, cls=True)
        except TypeError:
            result = _paddle.ocr(img)
        lines = _parse_paddle_result(result)
        return "\n".join(lines)
    except Exception as exc:
        logger.warning(f"PaddleOCR inference failed: {exc}")
        return ""


def _run_tesseract(img: np.ndarray) -> str:
    """Run Tesseract with multilingual support (English + French + Arabic)."""
    pil = Image.fromarray(img)
    return pytesseract.image_to_string(
        pil,
        lang="eng+fra+ara",
        config="--psm 6 --oem 3",
    )


# ---------------------------------------------------------------------------
# PDF handling
# ---------------------------------------------------------------------------

def _pdf_has_text_layer(pdf_bytes: bytes) -> bool:
    if not FITZ_AVAILABLE:
        return False
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in doc:
        if page.get_text("text").strip():
            doc.close()
            return True
    doc.close()
    return False


def _extract_pdf_text_layer(pdf_bytes: bytes) -> str:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    texts = [page.get_text("text") for page in doc]
    doc.close()
    return "\n\n".join(texts)


def _pdf_to_images(pdf_bytes: bytes) -> list[Image.Image]:
    if not PDF2IMAGE_AVAILABLE:
        raise RuntimeError(
            "pdf2image is not installed. "
            "Run: pip install pdf2image  (also needs system poppler-utils)."
        )
    return convert_from_bytes(pdf_bytes, dpi=200)


# ---------------------------------------------------------------------------
# Multi-pass OCR cross-validation helpers
# ---------------------------------------------------------------------------

def _word_set(text: str) -> set[str]:
    """Return a lowercase set of all ≥3-char word tokens found in text."""
    return set(re.findall(r"\b[\w\u0600-\u06FF]{3,}\b", text.lower()))


def _cross_validate_texts(texts: list[str]) -> tuple[str, float]:
    """
    Given OCR texts from multiple preprocessing passes, pick the best and
    compute a word-level confidence score.

    Strategy:
      - Primary  = longest non-empty result (carries the most information)
      - Confidence = fraction of primary words confirmed by ≥1 other pass

    Returns (consensus_text, confidence).  confidence ∈ [0.0, 1.0].
    """
    non_empty = [t for t in texts if t.strip()]
    if not non_empty:
        return "", 0.0
    if len(non_empty) == 1:
        logger.info("[OCR-MULTIPASS] Only 1 pass returned text — confidence capped at 0.55")
        return non_empty[0], 0.55

    primary = max(non_empty, key=len)
    others  = [t for t in non_empty if t is not primary]

    other_words: set[str] = set()
    for t in others:
        other_words |= _word_set(t)

    primary_words = re.findall(r"\b[\w\u0600-\u06FF]{3,}\b", primary.lower())
    if not primary_words:
        return primary, 0.55

    confirmed  = sum(1 for w in primary_words if w in other_words)
    confidence = round(confirmed / len(primary_words), 3)

    logger.info(
        "[OCR-MULTIPASS] passes=%d  primary_chars=%d  "
        "word_confidence=%.0f%%  (%d/%d words confirmed)",
        len(non_empty), len(primary), confidence * 100, confirmed, len(primary_words),
    )
    return primary, confidence


def _extract_suspicious_words(primary: str, other_words: set[str]) -> list[str]:
    """
    Return words from the primary OCR result that do NOT appear in any
    other pass — these are likely OCR misreads.
    Only considers words of ≥4 characters to skip noise.
    """
    tokens = re.findall(r"\b[\w\u0600-\u06FF]{4,}\b", primary.lower())
    seen:   set[str] = set()
    result: list[str] = []
    for w in tokens:
        if w not in other_words and w not in seen:
            seen.add(w)
            result.append(w)
    return result


# ---------------------------------------------------------------------------
# Core image handler (single-pass, kept for internal use)
# ---------------------------------------------------------------------------

def _handle_image_single(pil_image: Image.Image) -> str:
    """Run one OCR pass (Pass 1 only). Internal. Returns cleaned text."""
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")
    preprocessed = preprocess_image(pil_image)
    raw_text = _run_paddleocr(preprocessed)
    if raw_text.strip():
        logger.info("PaddleOCR extracted %d chars", len(raw_text))
    if not raw_text.strip():
        if TESSERACT_AVAILABLE:
            try:
                raw_text = _run_tesseract(preprocessed)
                if raw_text.strip():
                    logger.info("Tesseract extracted %d chars", len(raw_text))
            except Exception as exc:
                logger.error("Tesseract failed: %s", exc)
    if not raw_text.strip():
        raise RuntimeError(
            "OCR returned empty text. "
            "The image may be blank, too blurry, or contain no legible text."
        )
    return _clean_text(raw_text)


def _handle_image(pil_image: Image.Image) -> str:
    """
    Multi-pass OCR on a PIL image. Runs 3 preprocessing strategies and
    cross-validates results. Returns consensus cleaned text.

    Delegates to _handle_image_multipass internally; raises RuntimeError
    if all passes return empty text.
    """
    text, _conf, _susp = _handle_image_multipass(pil_image)
    return text


def _handle_image_multipass(pil_image: Image.Image) -> tuple[str, float, list[str]]:
    """
    Run 3 OCR preprocessing passes and cross-validate results.

    Returns:
        (consensus_text, ocr_confidence, suspicious_words)

    ocr_confidence ∈ [0.0, 1.0] = fraction of primary words confirmed by ≥1 other pass.
    suspicious_words = words present only in the primary pass (may be misread).
    """
    if pil_image.mode not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")

    texts: list[str] = []

    # ── Pass 0: Raw image (no preprocessing) ──────────────────────────────
    try:
        # Sometimes preprocessing ruins an already clean image
        t0 = _run_paddleocr(np.array(pil_image.convert("RGB")))
        if not t0.strip() and TESSERACT_AVAILABLE:
            t0 = _run_tesseract(np.array(pil_image.convert("RGB")))
        if t0.strip():
            texts.append(_clean_text(t0))
            logger.info("[OCR-PASS0] %d chars (raw)", len(t0))
        else:
            logger.warning("[OCR-PASS0] empty")
    except Exception as exc:
        logger.warning("[OCR-PASS0] failed: %s", exc)

    # ── Pass 1: Standard pipeline (binarize + CLAHE) ──────────────────────
    try:
        img1 = preprocess_image(pil_image)
        t1 = _run_paddleocr(img1)
        if not t1.strip() and TESSERACT_AVAILABLE:
            t1 = _run_tesseract(img1)
        if t1.strip():
            texts.append(_clean_text(t1))
            logger.info("[OCR-PASS1] %d chars", len(t1))
        else:
            logger.warning("[OCR-PASS1] empty")
    except Exception as exc:
        logger.warning("[OCR-PASS1] failed: %s", exc)

    # ── Pass 2: Soft enhance, no binarize ────────────────────────────────
    try:
        img2 = _preprocess_pass2(pil_image)
        t2 = _run_paddleocr(img2)
        if not t2.strip() and TESSERACT_AVAILABLE:
            t2 = _run_tesseract(img2)
        if t2.strip():
            texts.append(_clean_text(t2))
            logger.info("[OCR-PASS2] %d chars", len(t2))
        else:
            logger.warning("[OCR-PASS2] empty")
    except Exception as exc:
        logger.warning("[OCR-PASS2] failed: %s", exc)

    # ── Pass 3: Inverted binarize ─────────────────────────────────────────
    try:
        img3 = _preprocess_pass3(pil_image)
        t3 = _run_paddleocr(img3)
        if not t3.strip() and TESSERACT_AVAILABLE:
            t3 = _run_tesseract(img3)
        if t3.strip():
            texts.append(_clean_text(t3))
            logger.info("[OCR-PASS3] %d chars", len(t3))
        else:
            logger.warning("[OCR-PASS3] empty")
    except Exception as exc:
        logger.warning("[OCR-PASS3] failed: %s", exc)

    if not texts:
        raise RuntimeError(
            "OCR returned empty text on all 4 passes. "
            "The image may be blank, too blurry, or contain no legible text."
        )

    consensus_text, confidence = _cross_validate_texts(texts)

    # Build a set of words seen in passes other than the primary
    other_words: set[str] = set()
    for t in texts:
        if t is not consensus_text:
            other_words |= _word_set(t)

    suspicious = (
        _extract_suspicious_words(consensus_text, other_words)
        if other_words else []
    )

    return consensus_text, confidence, suspicious[:25]  # cap at 25 tokens


def _handle_pdf(pdf_bytes: bytes) -> str:
    if _pdf_has_text_layer(pdf_bytes):
        logger.info("PDF has native text layer — extracting directly.")
        return _clean_text(_extract_pdf_text_layer(pdf_bytes))

    logger.info("Scanned PDF — converting pages to images for OCR.")
    images = _pdf_to_images(pdf_bytes)
    page_texts: list[str] = []
    for i, img in enumerate(images):
        try:
            page_texts.append(_handle_image(img))
        except RuntimeError as exc:
            logger.warning(f"Page {i + 1} OCR failed: {exc} — skipped.")

    if not page_texts:
        raise RuntimeError("OCR returned empty text for all PDF pages.")

    return "\n\n".join(page_texts)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Extract text from an image or PDF file.

    Backward-compatible wrapper around extract_text_with_metadata().
    Returns only the text string.

    Args:
        file_bytes: Raw file content.
        filename:   Original filename (used to detect extension).

    Returns:
        Cleaned extracted text string.

    Raises:
        ValueError:   Unsupported file extension.
        RuntimeError: OCR produced no usable text.
    """
    return extract_text_with_metadata(file_bytes, filename)["text"]


def extract_text_with_metadata(file_bytes: bytes, filename: str) -> dict:
    """
    Extract text from an image or PDF with OCR quality metadata.

    Returns a dict:
    {
        "text":             str,         # cleaned OCR text
        "ocr_confidence":   float,       # 0.0–1.0 (word-level cross-pass agreement)
        "suspicious_words": list[str],   # tokens confirmed by only 1 pass (may be misread)
        "passes_used":      int,         # number of preprocessing passes that returned text
    }

    Raises:
        ValueError:   Unsupported file extension.
        RuntimeError: OCR produced no usable text.
    """
    suffix = Path(filename).suffix.lower()

    if suffix == ".pdf":
        text = _handle_pdf(file_bytes)
        return {
            "text":             text,
            "ocr_confidence":   1.0,   # PDFs with text layer are already clean
            "suspicious_words": [],
            "passes_used":      1,
        }

    if suffix in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}:
        pil_image = Image.open(io.BytesIO(file_bytes))
        text, confidence, suspicious = _handle_image_multipass(pil_image)
        return {
            "text":             text,
            "ocr_confidence":   confidence,
            "suspicious_words": suspicious,
            "passes_used":      3,
        }

    raise ValueError(
        f"Unsupported file type '{suffix}'. "
        "Accepted: PDF, JPG, PNG, BMP, TIFF, WEBP."
    )
