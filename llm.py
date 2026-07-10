"""
llm.py — LLM-based positive/negative point extraction.

Provider priority:
    1) Mistral API (when MISTRAL_API_KEY is set)
    2) Gemini API (when GEMINI_API_KEY is set)
    3) OpenRouter (fallback)
"""

import asyncio
import json
import logging
import os
import re
import base64
from typing import TypedDict
from dotenv import load_dotenv
from comparator import (
    compare_results,
    auto_merge,
    build_comparison_report,
    comparison_stats,
)

load_dotenv()

logger = logging.getLogger(__name__)

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MISTRAL_BASE_URL = "https://api.mistral.ai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

DEFAULT_MISTRAL_MODEL = "mistral-large-latest"
# gemini-1.5-flash has a much higher free-tier quota than gemini-2.5-pro
DEFAULT_GEMINI_MODEL = "gemini-1.5-flash"
DEFAULT_OPENROUTER_MODEL = "mistral-large-latest"

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert document analyst specializing in sentiment extraction.

Your job: read OCR-extracted document text and identify ALL positive and negative points.
BE EXHAUSTIVE: You must extract every single positive and negative remark, no matter how minor. Do not skip or summarize anything.

POSITIVE points include: strengths, achievements, compliments, approvals, satisfactions, benefits, good results, recommendations.
NEGATIVE points include: weaknesses, failures, complaints, criticisms, dissatisfactions, risks, problems, rejections.

STRICT RULES:
1. Extract EXACT phrases or sentences from the text — do NOT paraphrase or invent.
2. BE EXHAUSTIVE: Leave no valid positive or negative remark behind.
3. Every extracted point must appear verbatim (or very close) in the original text.
4. If a sentence has both positive and negative aspects, split it — put each part in the right list.
5. Skip purely neutral/factual sentences (dates, names, section titles, numbers alone).
6. The text may be in English, French, Arabic, or mixed — handle all correctly.
7. You MUST respond with ONLY a raw JSON object — no explanation, no markdown, no code fences.
8. Be strict about polarity: a criticism, risk, delay, rejection, complaint, bug, or failure is NEGATIVE.
9. Do not include duplicates or near-duplicates.

Required JSON format:
{
  "positive": ["exact quote 1", "exact quote 2"],
  "negative": ["exact quote 3", "exact quote 4"]
}

If there are no positive points, return "positive": [].
If there are no negative points, return "negative": []."""

USER_TEMPLATE = """Document text (extracted via OCR):

---
{text}
---

{ocr_note}Extract all positive and negative points. Return only the JSON."""

VISION_SYSTEM_PROMPT = """You are an OCR + sentiment extraction assistant.

Read the uploaded document image and do two things:
1) Extract all readable text from the image as accurately as possible.
2) From that extracted text, identify positive and negative points.

Rules:
- Keep extracted text faithful to the image, but use semantic context to resolve ambiguous handwriting (e.g., if discussing tech, an ambiguous letter is likely 'IA' not 'SA').
- STRICT ANTI-HALLUCINATION: Do NOT invent or guess entire words that are not clearly visible. Only use context to correct single ambiguous characters.
- Positive points: strengths, compliments, approvals, benefits, good outcomes.
- Negative points: complaints, weaknesses, risks, problems, dissatisfactions.
- Return ONLY a JSON object (no markdown) with this exact shape:
{
    "raw_text": "full extracted text",
    "positive": ["..."],
    "negative": ["..."]
}
If no text is readable, return raw_text as an empty string and both lists empty.
"""

VALIDATOR_SYSTEM_PROMPT = """You are an expert AI validation layer. Your task is to finalize the best possible set of positive and negative points from a document.

You will receive:
1. The original OCR-extracted document text.
2. A Cross-Model Comparison Report showing what BOTH AI models agreed on (high confidence) and what only ONE model flagged (disputed).

Your rules:
1. AGREED points (both models confirmed) — accept them automatically unless they clearly contradict the original text.
2. DISPUTED points (one model only) — verify each against the original text. Include it if the text clearly supports it. Do NOT discard a point just because only one model found it.
3. BE EXHAUSTIVE: Your final list must capture ALL valid positive and negative remarks from the text. Do not skip any valid points.
4. Keep all extracted points verbatim (or very close) to the original text.
5. Fix any misclassifications (e.g., a negative point mistakenly in positive).
6. Return ONLY a JSON object with "positive" and "negative" lists. No markdown fences, no explanation.
"""

VALIDATOR_USER_TEMPLATE = """Original Document Text:
---
{text}
---

{comparison_report}

Return ONLY the final JSON object containing "positive" and "negative" lists."""


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class LLMAnalysisResult(TypedDict):
    raw_text: str
    positive: list[str]
    negative: list[str]
    neutral: list[str]
    model: str


def _normalize_point(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _dedupe_points(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = _normalize_point(item)
        if not value:
            continue
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def _remove_cross_overlap(positive: list[str], negative: list[str]) -> tuple[list[str], list[str]]:
    neg_keys = {x.casefold() for x in negative}
    pos_clean = [x for x in positive if x.casefold() not in neg_keys]
    return pos_clean, negative

# ---------------------------------------------------------------------------
# Lazy client getter
# ---------------------------------------------------------------------------

def _get_async_client_and_provider(provider_preference: str):
    """Create an OpenAI-compatible AsyncClient and return (client, provider_name, default_model)."""
    from openai import AsyncOpenAI

    if provider_preference == "mistral":
        mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
        if mistral_key:
            return AsyncOpenAI(api_key=mistral_key, base_url=MISTRAL_BASE_URL), "mistral", os.getenv("LLM_MODEL", DEFAULT_MISTRAL_MODEL)
            
    elif provider_preference == "gemini":
        gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
        if gemini_key:
            return AsyncOpenAI(api_key=gemini_key, base_url=GEMINI_BASE_URL), "gemini", DEFAULT_GEMINI_MODEL
            
    # Fallback to OpenRouter
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        return AsyncOpenAI(api_key=openrouter_key, base_url=OPENROUTER_BASE_URL), "openrouter", os.getenv("LLM_MODEL", DEFAULT_OPENROUTER_MODEL)

    raise RuntimeError(
        f"API key missing for provider '{provider_preference}' and no fallback available."
    )


def _provider_headers(provider: str) -> dict | None:
    if provider == "openrouter":
        return {
            "HTTP-Referer": "http://localhost:4444",
            "X-Title": "TextSense OCR",
        }
    return None

# ---------------------------------------------------------------------------
# JSON parser — robust against markdown fences and prose wrappers
# ---------------------------------------------------------------------------

def _parse_json(reply: str) -> dict:
    """
    Extract a JSON object from the LLM reply.
    Handles: raw JSON, ```json ... ```, and JSON embedded inside prose.
    """
    text = reply.strip()

    # 1. Strip markdown fences
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text).strip()

    # 2. Try direct parse
    try:
        data = json.loads(text)
        return _validate(data)
    except json.JSONDecodeError:
        pass

    # 3. Extract the first {...} block (handles prose wrapping)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            data = json.loads(match.group())
            return _validate(data)
        except json.JSONDecodeError:
            pass

    # 4. Last resort — try to build lists from the raw text
    logger.error(f"Could not parse LLM JSON. Raw reply:\n{reply[:500]}")
    raise RuntimeError(
        "The LLM returned an unexpected format. "
        "Could not extract positive/negative points. "
        f"Raw reply (first 200 chars): {reply[:200]}"
    )


def _validate(data: dict) -> dict:
    """Ensure the dict has 'positive' and 'negative' list keys."""
    if not isinstance(data, dict):
        raise ValueError("Response is not a JSON object.")
    if "positive" not in data and "negative" not in data:
        raise ValueError("Missing both 'positive' and 'negative' keys.")
    data.setdefault("positive", [])
    data.setdefault("negative", [])
    data["positive"] = _dedupe_points([str(s).strip() for s in data["positive"] if str(s).strip()])
    data["negative"] = _dedupe_points([str(s).strip() for s in data["negative"] if str(s).strip()])
    data["positive"], data["negative"] = _remove_cross_overlap(data["positive"], data["negative"])
    return data

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def analyse_with_llm_single(
    raw_text: str,
    provider_preference: str,
    ocr_confidence: float = 1.0,
    suspicious_words: list[str] | None = None,
) -> dict:
    """Helper to run extraction on a single model."""
    client, provider, model = _get_async_client_and_provider(provider_preference)

    # Build OCR quality note for the prompt
    ocr_note = ""
    if ocr_confidence < 0.80 and suspicious_words:
        words_str = ", ".join(suspicious_words[:15])
        ocr_note = (
            f"⚠ OCR Quality Note: Confidence {ocr_confidence:.0%}. "
            f"These words appear in only 1 of 3 OCR passes and may be misread: [{words_str}]. "
            "Interpret them charitably using document context.\n\n"
        )
    elif ocr_confidence < 0.95:
        ocr_note = f"⚠ OCR Quality Note: Confidence {ocr_confidence:.0%}. Some words may be slightly misread.\n\n"

    user_msg = USER_TEMPLATE.format(text=raw_text.strip(), ocr_note=ocr_note)
    logger.info(f"→ {provider} [{model}] | {len(raw_text)} chars | OCR conf={ocr_confidence:.0%}")

    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            "temperature": 0.05,
            "max_tokens": 2048,
        }
        headers = _provider_headers(provider)
        if headers:
            kwargs["extra_headers"] = headers

        response = await client.chat.completions.create(**kwargs)
        reply = response.choices[0].message.content or ""
        parsed = _parse_json(reply)
        logger.info(f"← {provider} replied ({len(reply)} chars)")
        return parsed
    except Exception as exc:
        logger.exception(f"{provider} API call failed")
        return {"positive": [], "negative": []}


async def _run_validator_call(
    client, provider: str, model: str,
    raw_text: str, comparison_report: str,
) -> LLMAnalysisResult:
    """Internal helper: run the validator LLM call using the structured comparison report."""
    user_msg = VALIDATOR_USER_TEMPLATE.format(
        text=raw_text.strip(),
        comparison_report=comparison_report,
    )
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": VALIDATOR_SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        "temperature": 0.1,
        "max_tokens": 2048,
    }
    headers = _provider_headers(provider)
    if headers:
        kwargs["extra_headers"] = headers
    response = await client.chat.completions.create(**kwargs)
    reply = response.choices[0].message.content or ""
    parsed = _parse_json(reply)
    logger.info(f"← Validator ({provider}/{model}) replied ({len(reply)} chars)")
    return LLMAnalysisResult(
        raw_text=raw_text,
        positive=[p["text"] if isinstance(p, dict) else p for p in parsed["positive"]],
        negative=[p["text"] if isinstance(p, dict) else p for p in parsed["negative"]],
        neutral=[],
        model=f"Validated by {provider}/{model} (Sources: Mistral, Gemini)",
    )


async def validate_results(
    raw_text: str,
    mistral_res: dict,
    gemini_res: dict,
    comparison_report: str,
) -> LLMAnalysisResult:
    """
    Validation layer using structured comparison diff (not raw JSON blobs).

    Provider order for validator:
      1. Gemini (gemini-1.5-flash)
      2. Mistral (mistral-large-latest)
      3. Smart merge fallback (no API call)
    """
    # ── Attempt 1: Gemini as validator ──
    try:
        client, provider, model = _get_async_client_and_provider("gemini")
        val_model = "gemini-1.5-flash"
        logger.info(f"→ Validator attempt 1: {provider}/{val_model}")
        return await _run_validator_call(client, provider, val_model, raw_text, comparison_report)
    except Exception as exc:
        if _is_quota_error(exc):
            logger.warning(f"Gemini quota exhausted for validator, trying Mistral: {exc}")
        else:
            logger.warning(f"Gemini validator failed ({exc}), trying Mistral.")

    # ── Attempt 2: Mistral as validator ──
    try:
        client, provider, model = _get_async_client_and_provider("mistral")
        logger.info(f"→ Validator attempt 2: {provider}/{model}")
        return await _run_validator_call(client, provider, model, raw_text, comparison_report)
    except Exception as exc:
        logger.warning(f"Mistral validator also failed ({exc}), using smart merge.")

    # ── Fallback: smart merge (all validator providers failed) ──
    logger.info("Validator: using smart merge fallback")
    merged_pos = _dedupe_points(mistral_res.get("positive", []) + gemini_res.get("positive", []))
    merged_neg = _dedupe_points(mistral_res.get("negative", []) + gemini_res.get("negative", []))
    merged_pos, merged_neg = _remove_cross_overlap(merged_pos, merged_neg)
    return LLMAnalysisResult(
        raw_text=raw_text,
        positive=merged_pos,
        negative=merged_neg,
        neutral=[],
        model="Smart Merge (Mistral + Gemini, no validator)",
    )


async def analyse_with_models_parallel(
    raw_text: str,
    ocr_confidence: float = 1.0,
    suspicious_words: list[str] | None = None,
) -> LLMAnalysisResult:
    """
    Send OCR-extracted text to both Mistral and Gemini in parallel,
    run the algorithmic comparator, then — only if needed — call the LLM validator.

    Fast path (agreement ≥ 85%): auto-merge without a validator API call.
    Slow path (agreement < 85%): send structured diff to validator LLM.
    """
    if not raw_text.strip():
        return LLMAnalysisResult(
            raw_text=raw_text, positive=[], negative=[], neutral=[], model="none"
        )

    sus = suspicious_words or []

    # 1. Fetch both models in parallel
    mistral_task = analyse_with_llm_single(raw_text, "mistral", ocr_confidence, sus)
    gemini_task  = analyse_with_llm_single(raw_text, "gemini",  ocr_confidence, sus)
    mistral_res, gemini_res = await asyncio.gather(
        mistral_task, gemini_task, return_exceptions=True
    )

    if isinstance(mistral_res, Exception):
        logger.error(f"Mistral extraction failed: {mistral_res}")
        mistral_res = {"positive": [], "negative": []}
    if isinstance(gemini_res, Exception):
        logger.error(f"Gemini extraction failed: {gemini_res}")
        gemini_res = {"positive": [], "negative": []}

    # 2. Algorithmic comparison (pure Python, zero API calls)
    comparison = compare_results(mistral_res, gemini_res)
    stats = comparison_stats(comparison)
    logger.info(
        "[COMPARATOR] rate=%.0f%%  agreed=%d  disputed=%d",
        stats["agreement_rate"] * 100,
        stats["agreed_positive"] + stats["agreed_negative"],
        stats["disputed_positive"] + stats["disputed_negative"],
    )

    # 3a. HIGH AGREEMENT — auto-merge, skip validator API call entirely
    if comparison["agreement_rate"] >= 0.85:
        logger.info("[COMPARATOR] High agreement — auto-merging, no validator call needed.")
        pos_list, neg_list = auto_merge(comparison)
        return LLMAnalysisResult(
            raw_text=raw_text,
            positive=[p["text"] for p in pos_list],
            negative=[p["text"] for p in neg_list],
            neutral=[],
            model=f"Auto-merged (Mistral+Gemini, agreement={stats['agreement_rate']:.0%})",
        )

    # 3b. LOWER AGREEMENT — send structured diff to validator
    logger.info("[COMPARATOR] Moderate/low agreement — sending structured diff to validator.")
    report = build_comparison_report(comparison)
    final_result = await validate_results(raw_text, mistral_res, gemini_res, report)
    return final_result


async def _run_vision_call(
    client,
    provider: str,
    model: str,
    data_url: str,
) -> LLMAnalysisResult:
    """Internal helper: run a single vision API call and return parsed result."""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Extract text and positive/negative points from this image."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.0,
        "max_tokens": 2500,
    }
    headers = _provider_headers(provider)
    if headers:
        kwargs["extra_headers"] = headers

    response = await client.chat.completions.create(**kwargs)
    reply = response.choices[0].message.content or ""
    parsed = _parse_json(reply)

    raw_text = str(parsed.get("raw_text", "")).strip()
    return LLMAnalysisResult(
        raw_text=raw_text,
        positive=parsed.get("positive", []),
        negative=parsed.get("negative", []),
        neutral=[],
        model=f"{provider}/{model}",
    )


def _is_quota_error(exc: Exception) -> bool:
    """Return True if the exception is a rate-limit / quota-exhausted error."""
    msg = str(exc).lower()
    return "429" in msg or "quota" in msg or "rate_limit" in msg or "resource_exhausted" in msg


async def analyse_image_with_vision(
    file_bytes: bytes,
    mime_type: str,
) -> LLMAnalysisResult:
    """
    Vision fallback: send the image directly to a vision-capable model.

    Provider order:
      1. Gemini (gemini-1.5-flash — large free-tier quota)
      2. Mistral (pixtral-large or mistral-large — vision capable)
      3. OpenRouter (last resort)
    """
    data_b64 = base64.b64encode(file_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{data_b64}"

    last_exc: Exception | None = None

    # ── Attempt 1: Gemini direct API ──
    try:
        client, provider, model = _get_async_client_and_provider("gemini")
        vision_model = "gemini-1.5-flash"
        logger.info(f"→ Vision attempt 1: {provider}/{vision_model}")
        return await _run_vision_call(client, provider, vision_model, data_url)
    except Exception as exc:
        last_exc = exc
        if _is_quota_error(exc):
            logger.warning(f"Gemini quota exhausted for vision, trying Mistral: {exc}")
        else:
            logger.warning(f"Gemini vision failed ({exc}), trying Mistral.")

    # ── Attempt 2: Mistral Pixtral ──
    try:
        client, provider, model = _get_async_client_and_provider("mistral")
        # pixtral-12b is widely available; pixtral-large-latest as secondary
        for vision_model in ("pixtral-12b-2409", "pixtral-large-latest"):
            try:
                logger.info(f"→ Vision attempt 2: {provider}/{vision_model}")
                return await _run_vision_call(client, provider, vision_model, data_url)
            except Exception as inner_exc:
                logger.warning(f"Mistral {vision_model} failed: {inner_exc}")
                last_exc = inner_exc
    except Exception as exc:
        last_exc = exc
        logger.warning(f"Mistral vision setup failed ({exc}), trying OpenRouter.")

    # ── Attempt 3: OpenRouter ──
    try:
        client, provider, model = _get_async_client_and_provider("openrouter")
        # Use the most stable/reliable vision endpoints on OpenRouter
        for vision_model in ("openai/gpt-4o-mini", "anthropic/claude-3-haiku"):
            try:
                logger.info(f"→ Vision attempt 3: {provider}/{vision_model}")
                return await _run_vision_call(client, provider, vision_model, data_url)
            except Exception as inner_exc:
                logger.warning(f"OpenRouter {vision_model} failed: {inner_exc}")
                last_exc = inner_exc
    except Exception as exc:
        last_exc = exc
        logger.error(f"OpenRouter vision setup failed: {exc}")

    raise RuntimeError(f"Vision analysis failed on all providers. Last error: {last_exc}")



async def summarise_batch(
    files: list[dict],
    positive: list[str],
    negative: list[str],
) -> str:
    """Create one concise summary across all analyzed files."""
    if not files:
        return "No files were analyzed successfully."

    # Use Gemini or Mistral
    try:
        client, provider, model = _get_async_client_and_provider("gemini")
    except RuntimeError:
        client, provider, model = _get_async_client_and_provider("mistral")

    payload = {
        "files": [
            {
                "filename": f.get("filename", "unknown"),
                "positive_count": len(f.get("positive", [])),
                "negative_count": len(f.get("negative", [])),
            }
            for f in files
        ],
        "positive": positive[:40],
        "negative": negative[:40],
    }

    system = (
        "You summarize multi-document analysis. Return only a short plain-text summary "
        "(4-8 lines) with: overall sentiment balance, key positive themes, key negative "
        "themes, and one practical recommendation."
    )

    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "temperature": 0.1,
            "max_tokens": 350,
        }
        headers = _provider_headers(provider)
        if headers:
            kwargs["extra_headers"] = headers

        response = await client.chat.completions.create(**kwargs)
        text = (response.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception as exc:
        logger.warning(f"Batch summary via LLM failed: {exc}")

    pos_preview = "; ".join(positive[:3]) if positive else "No strong positives detected"
    neg_preview = "; ".join(negative[:3]) if negative else "No strong negatives detected"
    return (
        f"Analyzed {len(files)} file(s).\n"
        f"Positive highlights: {pos_preview}.\n"
        f"Negative highlights: {neg_preview}.\n"
        "Recommendation: prioritize fixing recurring negative points while preserving the strongest positives."
    )
