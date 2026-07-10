 # TextSense OCR

> Extract text from images & PDFs with AI-powered OCR, then classify every sentence as **positive**, **negative**, or **neutral** using multilingual sentiment analysis.

Supports **English**, **French**, and **Arabic** documents.

---

## Architecture

```
TextSenseOCR/
├── main.py          # FastAPI app — POST /analyze endpoint
├── ocr.py           # OCR pipeline (preprocessing → PaddleOCR → Tesseract fallback)
├── sentiment.py     # Sentence splitting + HuggingFace sentiment classification
├── static/
│   └── index.html   # Single-page frontend (drag-and-drop, results view)
├── requirements.txt
└── README.md
```

### Models used

| Component | Model / Library |
|---|---|
| OCR (primary) | PaddleOCR (`use_angle_cls=True`) |
| OCR (fallback) | Tesseract via pytesseract (`eng+fra+ara`) |
| PDF text layer | PyMuPDF (`fitz`) |
| Sentiment | `cardiffnlp/twitter-xlm-roberta-base-sentiment` (HuggingFace) |
| Sentence splitting | spaCy sentencizer (`xx` blank model) / NLTK fallback |

---

## System Dependencies

These must be installed **before** running `pip install`.

### Linux / WSL
```bash
# Tesseract OCR + language packs
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-fra tesseract-ocr-ara

# Poppler (for pdf2image / PDF → image conversion)
sudo apt-get install -y poppler-utils
```

### macOS
```bash
brew install tesseract tesseract-lang poppler
```

### Windows
1. Download and install [Tesseract for Windows](https://github.com/UB-Mannheim/tesseract/wiki) (include French + Arabic language data during install).
2. Download [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/) and add its `bin/` folder to your `PATH`.
3. Ensure `tesseract` is on your `PATH` (or set `pytesseract.pytesseract.tesseract_cmd` in `ocr.py`).

---

## Python Setup

### 1. Create a virtual environment (recommended)
```bash
python -m venv .venv

# Activate
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

> **GPU users:** Replace the `torch==2.3.0` line in `requirements.txt` with the appropriate CUDA wheel from [pytorch.org](https://pytorch.org/get-started/locally/).

### 3. Download the spaCy multilingual model (optional but recommended)
The code uses the lightweight blank `xx` model by default (no download needed).
If you want better sentence boundary detection, install a larger model:
```bash
python -m spacy download xx_ent_wiki_sm
```

### 4. NLTK data (auto-downloaded on first run)
`punkt` and `punkt_tab` are downloaded automatically if NLTK is used as the fallback.

---

## Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

The `--reload` flag enables auto-restart on code changes (development mode). Remove it for production.

---

## API Reference

### `POST /analyze`

Upload a file and receive extracted text + sentiment buckets.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | binary | Image (jpg, png, bmp, tiff, webp) or PDF |

**Response** — `application/json`

```json
{
  "raw_text": "Full extracted text from the document...",
  "positive": ["Great product, really satisfied!", "..."],
  "negative": ["This was a terrible experience.", "..."],
  "neutral":  ["The report was submitted on Monday.", "..."]
}
```

**Error codes**

| Code | Meaning |
|---|---|
| 413 | File exceeds 20 MB limit |
| 415 | Unsupported file type |
| 422 | OCR returned empty text (blank/unreadable image) |
| 500 | Unexpected server error |

---

## Advanced architecture (what makes this pipeline non-classical)

This project goes beyond “OCR → one sentiment model”. It implements a **hybrid OCR layer**, then a **two-agent parallel extraction**, then a **scoring comparator** with an optional **validator layer**.

### 1) Hybrid OCR: PDF native text vs scanned raster → multi-pass preprocessing
- **PDFs**: uses **PyMuPDF (`fitz`)** to detect and extract the **native text layer** (`page.get_text("text")`).
- **Scanned PDFs**: converts pages to images using **pdf2image** (Poppler dependency), then runs OCR page-by-page.
- **Images**: operates on uploaded **binary bytes** decoded in-memory via **Pillow**, then processed with **OpenCV**.

#### Multi-pass OCR + OCR quality scoring
For images, the pipeline runs multiple OpenCV preprocessing strategies:
- grayscale
- deskew via **Canny + HoughLinesP**
- denoise via **fastNlMeansDenoising** (median blur fallback)
- binarize via **Otsu** or **adaptive Gaussian threshold**
- optional 2× **bicubic upscale**
- **CLAHE** contrast enhancement
- plus an additional **inverted binarization** pass for dark-background documents

It computes an **`ocr_confidence`** using **word-level agreement** across passes (confirmed ≥3-char tokens) and also emits **`suspicious_words`** (tokens present in only one pass). This score is injected into the LLM prompt as a quality note.

### 2) Parallel “agents”: Mistral + Gemini run concurrently
The extraction stage is implemented as two independent LLM hypotheses executed in parallel using **`asyncio.gather()`**:
- **Mistral agent**
- **Gemini agent**

Both agents are forced to be **exhaustive** and output **JSON only** containing:
- `positive`: verbiatim-or-close points extracted from OCR text
- `negative`: verbiatim-or-close points extracted from OCR text

### 3) Deterministic scoring comparator (SequenceMatcher fuzzy agreement)
Instead of trusting one model, the repo runs a **pure-Python comparator** on the two lists:
- Similarity computed with **`difflib.SequenceMatcher`** on normalized text.
- Points are bucketed into **AGREED** vs **DISPUTED** using explicit thresholds.
- A global **`agreement_rate`** is computed.

### 4) Validation layer: judge-by-structured-diff (only when needed)
If `agreement_rate` is below the acceptance threshold, the system calls a **validator LLM**.

Crucially, the validator is not given raw two blobs. It receives:
- the original OCR text
- and a **structured comparison report** enumerating AGREED and DISPUTED points (with confidence + source)

This “structured diff” strategy makes the validator behave like a constrained judge that keeps what both agreed on and verifies what only one flagged.

### 5) Local vs cloud operation
- **Local**: OCR (PaddleOCR + OpenCV + Tesseract fallback) and **HuggingFace** sentiment fallback.
- **Cloud**: extraction/validation via provider keys (**Mistral/Gemini/OpenRouter**) using an OpenAI-compatible async client.

> Note: the repo does not implement offset-based “extract specific bytes” forensic carving; it uses byte-level in-memory handling and semantic extraction (PDF text layer via PyMuPDF; scanned PDFs via rasterization; images via decoding + preprocessing).

---

## OCR Preprocessing Pipeline


Applied to every image before OCR (in order):

1. **Grayscale** — reduce colour noise
2. **Deskew** — detect and correct skew angle via Hough lines
3. **Denoise** — `fastNlMeansDenoising` (median blur fallback)
4. **Binarize** — Otsu global + adaptive Gaussian threshold, combined
5. **Upscale** — 2× bicubic upscale if width < 1000 px
6. **CLAHE** — contrast-limited adaptive histogram equalisation

---

## Notes & Limitations

- **PaddleOCR Arabic** — Arabic OCR quality depends on font clarity; handwriting is not reliably supported.
- **Sentiment on very short sentences** — fragments under 10 characters are labelled `neutral` without inference.
- **GPU acceleration** — Install a CUDA-enabled PyTorch wheel to speed up the sentiment model on large documents.
- **File size limit** — Hard-coded at 20 MB; change `MAX_FILE_SIZE_MB` in `main.py` if needed.

