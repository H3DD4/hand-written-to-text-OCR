# TextSense OCR — Tech Architecture (Parallel Agents + Comparator Validation)

This repository is not a “classic OCR + one LLM” pipeline. It is built as a **hybrid OCR** system followed by a **multi-hypothesis extraction** stage, then a **scoring comparator** and an optional **validator layer** that enforces exhaustiveness.

---

## 1) OCR Core: hybrid extraction + multi-pass cross-validation

### PDF: native text layer when available (PyMuPDF)
- **Engine**: `PyMuPDF` (`fitz`)
- **Logic**:
  - Detects whether pages contain a native text layer via `page.get_text("text")`.
  - If text exists: extracts directly (no OCR engine needed).
  - If text does not exist: treats the PDF as scanned.

### Scanned PDFs: rasterize → OCR pages
- **Engine**: `pdf2image`
- **Dependency**: system **Poppler** (required outside Python)
- **Logic**:
  - `convert_from_bytes(pdf_bytes, dpi=200)` produces PIL images.
  - Each page image is OCR’d and then concatenated.

### Images: in-memory byte-driven decoding + preprocessing
- The server receives uploaded content as raw **binary bytes**.
- Images are decoded via `PIL.Image.open(io.BytesIO(file_bytes))` and then processed locally.

### Multi-pass OCR preprocessing (OpenCV)
For image OCR, the system runs **multiple preprocessing strategies** and cross-validates the results.

**OpenCV transforms used** (see `ocr.py`):
1. grayscale conversion
2. deskew using **Canny + HoughLinesP**
3. denoise using **fastNlMeansDenoising** (fallback to median blur)
4. binarization using **Otsu** or **adaptive Gaussian threshold**
5. upscale (2× bicubic when width is below threshold)
6. CLAHE (`cv2.createCLAHE`)

A fourth pass uses **inverted binarization** to handle dark-background or white-text documents.

### OCR “confidence” scoring (word-level agreement)
Instead of trusting one preprocessing result, the repo computes a quantitative score:
- It runs 4 passes (Pass0 raw image + Pass1/2/3 preprocessing variants).
- Chooses a **primary** OCR text (longest non-empty result).
- Builds word token sets (≥3 chars) from the other passes.
- Computes:
  - `ocr_confidence = confirmed_primary_words / total_primary_words`
- Also extracts `suspicious_words`: tokens present only in the primary pass.

This scoring becomes an **input quality note** to the LLM extraction prompt.

---

## 2) Cloud-or-local hybrid agents: extraction with two concurrent LLM hypotheses

### “Agents” are parallel hypotheses, not threads of reasoning
The multi-agent behavior is implemented as two **independent extraction agents** running concurrently.

In `llm.py`:
- `analyse_with_models_parallel()` launches two coroutines simultaneously using **`asyncio.gather()`**:
  1) **Mistral agent**: `provider_preference="mistral"`
  2) **Gemini agent**: `provider_preference="gemini"`

Each agent:
- receives the same OCR text (plus an OCR quality note derived from `ocr_confidence` + `suspicious_words`)
- is prompted to be **exhaustive** and to output **only JSON** with:
  - `positive`: list of exact phrases (verbatim or near-verbatim)
  - `negative`: list of exact phrases

### Provider flexibility (local fallback exists)
LLM calls are cloud-based when API keys are present:
- `MISTRAL_API_KEY`
- `GEMINI_API_KEY`
- `OPENROUTER_API_KEY` (fallback)

If LLM extraction fails entirely, the pipeline falls back to **local sentiment classification**:
- `sentiment.py` uses HuggingFace `transformers` + PyTorch `torch`
- model: `cardiffnlp/twitter-xlm-roberta-base-sentiment`
- sentence splitting: spaCy blank `xx` sentencizer (fallback to NLTK)

So the system supports **cloud agents** and also a **local operational mode**.

---

## 3) Scoring system for agreement: comparator with fuzzy string matching

After both LLM agents return points, the repo does not “pick the bigger list”. It scores overlap with deterministic logic.

In `comparator.py`:

### Matching technique
- **`difflib.SequenceMatcher`**
- similarity is computed on normalized strings:
  - `casefold()`
  - whitespace collapsing

### Confidence bucketing
For each point produced by one model, the comparator searches best matches in the other model’s set:
- **AGREED** if similarity ≥ `AGREE_THRESHOLD` (`0.82`)
- **DISPUTED near-match** if similarity ≥ `NEAR_MATCH_THRESHOLD` (`0.55`)
- otherwise considered **unique / one-model-only** with baseline confidence.

### Global agreement score
The comparator computes:
- `agreement_rate` (ratio of paired agreed matches over total seen points)

This numeric score becomes the gate:
- High agreement ⇒ skip validator
- Moderate/low agreement ⇒ call validator LLM

---

## 4) Validation layer: structured diff + validator LLM judge

The validator is a second-stage LLM call that is explicitly constrained.

In `llm.py`:
1) `compare_results()` produces an internal structured comparison.
2) `build_comparison_report()` converts it into a **human-readable diff artifact** containing:
   - `✓ AGREED Positive` points
   - `✓ AGREED Negative` points
   - `⚠ DISPUTED` points with source and confidence
3) If `agreement_rate < 0.85`:
   - the system calls `_run_validator_call()`.

### Validator prompt constraints
The validator prompt enforces:
- keep **AGREED** points automatically unless they contradict OCR text
- verify **DISPUTED** points against original OCR text
- repair misclassification (positive vs negative)
- stay exhaustive and return only JSON

Provider order for validator:
- Gemini validator attempt first
- then Mistral
- then smart merge fallback if validators fail

---

## 5) Why this looks “non-classical OCR”

Unlike pipelines that:
- OCR → send everything to one LLM
- or OCR → sentiment model only

this repo implements a multi-stage, score-driven system:

1) **Hybrid OCR**
   - PDF native text layer vs scanned raster OCR
   - multi-pass preprocessing and **word-level confidence**

2) **Parallel multi-agent extraction**
   - simultaneous Mistral+Gemini JSON-only extraction

3) **Deterministic scoring + agreement rate**
   - fuzzy matching via SequenceMatcher with explicit thresholds

4) **Validation layer (judge-by-structured-diff)**
   - validator LLM is not fed raw outputs; it receives a structured agreement/dispute report

5) **Smart merge fast path**
   - skips validation when agreement is high

---

## Note on “byte carving”

The code does not implement forensic “extract specific bytes at offsets” for decoding binary formats.

The hybrid is at the **content-extraction layer**:
- PDF native text streams via PyMuPDF
- scanned PDFs converted to images
- uploaded binaries decoded to images in-memory

If you want true byte-offset carving, that would require a new module (file-signature scanning + parser-specific extraction).

