"""
comparator.py — Algorithmic comparison engine for Mistral vs Gemini results.

Uses difflib.SequenceMatcher (Python stdlib) — zero extra dependencies.

Flow:
  1. compare_results(mistral_res, gemini_res)  →  ComparisonResult
  2. auto_merge(comparison)                    →  (positive, negative) lists with confidence
  3. build_comparison_report(comparison)       →  structured diff string for the LLM validator

Confidence scale:
  0.91–1.00  → both models agreed  (AGREED zone)
  0.55–0.90  → one model only, near-match exists  (DISPUTED zone)
  0.50–0.60  → one model only, no match  (UNIQUE zone)
"""

import re
import logging
from difflib import SequenceMatcher
from typing import TypedDict

logger = logging.getLogger(__name__)

# ── Similarity thresholds ────────────────────────────────────────────────────
AGREE_THRESHOLD      = 0.82   # above → both models agree → auto-include at high confidence
NEAR_MATCH_THRESHOLD = 0.55   # above but below AGREE → disputed, send to validator
LOW_CONFIDENCE       = 0.60   # baseline confidence for model-unique points


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class PointWithConfidence(TypedDict):
    text:       str
    confidence: float   # 0.0–1.0
    category:   str     # "positive" | "negative"
    source:     str     # "both" | "mistral" | "gemini"


class ComparisonResult(TypedDict):
    agreed_positive:   list[PointWithConfidence]
    agreed_negative:   list[PointWithConfidence]
    disputed_positive: list[PointWithConfidence]
    disputed_negative: list[PointWithConfidence]
    agreement_rate:    float   # 0.0–1.0
    total_points:      int


# ---------------------------------------------------------------------------
# Core similarity helpers
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase + collapse whitespace for stable comparison."""
    return re.sub(r"\s+", " ", str(text or "").casefold().strip())


def fuzzy_similarity(a: str, b: str) -> float:
    """
    Return a 0.0–1.0 similarity score between two strings.
    Uses SequenceMatcher on normalized forms for language-agnostic matching.
    """
    a_n = _normalize(a)
    b_n = _normalize(b)
    if not a_n and not b_n:
        return 1.0
    if not a_n or not b_n:
        return 0.0
    return SequenceMatcher(None, a_n, b_n).ratio()


def _best_match(point: str, candidates: list[str]) -> tuple[float, int, str]:
    """
    Find the highest-similarity candidate for `point`.
    Returns (score, index, matched_text). index = -1 if no candidates.
    """
    best_score = 0.0
    best_idx   = -1
    best_text  = ""
    for i, candidate in enumerate(candidates):
        score = fuzzy_similarity(point, candidate)
        if score > best_score:
            best_score = score
            best_idx   = i
            best_text  = candidate
    return best_score, best_idx, best_text


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

def compare_results(
    mistral_res: dict,
    gemini_res:  dict,
) -> ComparisonResult:
    """
    Compare Mistral and Gemini extraction results point-by-point.

    For each Mistral point:
      - Find the best fuzzy-matching Gemini point
      - If similarity >= AGREE_THRESHOLD   → AGREED (high confidence)
      - If similarity >= NEAR_MATCH_THRESHOLD → DISPUTED near-match
      - Otherwise                          → UNIQUE (one model only)

    Unmatched Gemini points → DISPUTED or UNIQUE.

    Returns a ComparisonResult with four buckets + global agreement_rate.
    """
    agreed_pos:   list[PointWithConfidence] = []
    agreed_neg:   list[PointWithConfidence] = []
    disputed_pos: list[PointWithConfidence] = []
    disputed_neg: list[PointWithConfidence] = []

    total_agreed = 0
    total_seen   = 0

    for category in ("positive", "negative"):
        mistral_pts: list[str] = list(mistral_res.get(category) or [])
        gemini_pts:  list[str] = list(gemini_res.get(category)  or [])

        agreed_target   = agreed_pos   if category == "positive" else agreed_neg
        disputed_target = disputed_pos if category == "positive" else disputed_neg

        total_seen += len(mistral_pts) + len(gemini_pts)

        gemini_matched: set[int] = set()   # indices in gemini_pts already paired

        for m_point in mistral_pts:
            score, g_idx, g_text = _best_match(m_point, gemini_pts)

            if score >= AGREE_THRESHOLD:
                # ── Both models agree ──────────────────────────────────────
                # Prefer the longer/more detailed version of the two
                final_text = m_point if len(m_point) >= len(g_text) else g_text
                agreed_target.append(PointWithConfidence(
                    text=final_text,
                    confidence=round((1.0 + score) / 2, 3),   # maps to 0.91–1.0
                    category=category,
                    source="both",
                ))
                if g_idx >= 0:
                    gemini_matched.add(g_idx)
                total_agreed += 2

            elif score >= NEAR_MATCH_THRESHOLD:
                # ── Near-match — disputed ──────────────────────────────────
                disputed_target.append(PointWithConfidence(
                    text=m_point,
                    confidence=round(score * 0.85, 3),
                    category=category,
                    source="mistral",
                ))

            else:
                # ── Mistral-only point ─────────────────────────────────────
                disputed_target.append(PointWithConfidence(
                    text=m_point,
                    confidence=LOW_CONFIDENCE,
                    category=category,
                    source="mistral",
                ))

        # Add all unmatched Gemini points to disputed
        for i, g_point in enumerate(gemini_pts):
            if i in gemini_matched:
                continue
            score, _, _ = _best_match(g_point, mistral_pts)
            confidence = (
                round(score * 0.85, 3)
                if score >= NEAR_MATCH_THRESHOLD
                else LOW_CONFIDENCE
            )
            disputed_target.append(PointWithConfidence(
                text=g_point,
                confidence=confidence,
                category=category,
                source="gemini",
            ))

    agreement_rate = round(total_agreed / max(total_seen, 1), 3)

    logger.info(
        "[COMPARATOR] agreed=%d  disputed=%d  rate=%.0f%%",
        len(agreed_pos) + len(agreed_neg),
        len(disputed_pos) + len(disputed_neg),
        agreement_rate * 100,
    )

    return ComparisonResult(
        agreed_positive=agreed_pos,
        agreed_negative=agreed_neg,
        disputed_positive=disputed_pos,
        disputed_negative=disputed_neg,
        agreement_rate=agreement_rate,
        total_points=total_seen,
    )


# ---------------------------------------------------------------------------
# Auto-merge
# ---------------------------------------------------------------------------

def auto_merge(
    comparison: ComparisonResult,
) -> tuple[list[dict], list[dict]]:
    """
    Merge agreed + disputed points into final (positive, negative) lists.

    Returns two lists of dicts:
        [{"text": str, "confidence": float, "source": str}, ...]

    Agreed points always come first (sorted by confidence desc).
    Duplicates and cross-category overlaps are removed.
    """
    def _dedup(points: list[PointWithConfidence]) -> list[dict]:
        seen: set[str] = set()
        out:  list[dict] = []
        for p in sorted(points, key=lambda x: -x["confidence"]):
            key = _normalize(p["text"])
            if not key or key in seen:
                continue
            seen.add(key)
            out.append({
                "text":       p["text"],
                "confidence": p["confidence"],
                "source":     p["source"],
            })
        return out

    positive = _dedup(comparison["agreed_positive"] + comparison["disputed_positive"])
    negative = _dedup(comparison["agreed_negative"] + comparison["disputed_negative"])

    # Remove cross-category overlaps (same text cannot be both positive & negative)
    neg_keys = {_normalize(p["text"]) for p in negative}
    positive = [p for p in positive if _normalize(p["text"]) not in neg_keys]

    return positive, negative


# ---------------------------------------------------------------------------
# Structured diff report for the LLM validator
# ---------------------------------------------------------------------------

def build_comparison_report(comparison: ComparisonResult) -> str:
    """
    Build a human-readable structured diff for the LLM validator.

    Instead of sending raw JSON blobs, the validator receives a precise
    breakdown: what both models agreed on, and only what's disputed.
    This makes the LLM decision sharper, faster, and more accurate.
    """
    lines: list[str] = [
        "## Cross-Model Comparison Report",
        f"Agreement rate: {comparison['agreement_rate']:.0%}  |  "
        f"Total extraction points seen: {comparison['total_points']}",
        "",
    ]

    if comparison["agreed_positive"]:
        lines.append("### ✓ AGREED Positive — both models confirmed these:")
        for p in comparison["agreed_positive"]:
            lines.append(f"  [conf={p['confidence']:.2f}] {p['text']}")
        lines.append("")

    if comparison["agreed_negative"]:
        lines.append("### ✓ AGREED Negative — both models confirmed these:")
        for p in comparison["agreed_negative"]:
            lines.append(f"  [conf={p['confidence']:.2f}] {p['text']}")
        lines.append("")

    if comparison["disputed_positive"]:
        lines.append("### ⚠ DISPUTED Positive — only one model flagged these (verify against original text):")
        for p in comparison["disputed_positive"]:
            lines.append(f"  [source={p['source']}, conf={p['confidence']:.2f}] {p['text']}")
        lines.append("")

    if comparison["disputed_negative"]:
        lines.append("### ⚠ DISPUTED Negative — only one model flagged these (verify against original text):")
        for p in comparison["disputed_negative"]:
            lines.append(f"  [source={p['source']}, conf={p['confidence']:.2f}] {p['text']}")
        lines.append("")

    if comparison["agreement_rate"] >= 0.85:
        lines.append(
            "HIGH AGREEMENT: Both models largely agree. "
            "Accept AGREED points as-is. Review DISPUTED points carefully against the original text."
        )
    elif comparison["agreement_rate"] >= 0.50:
        lines.append(
            "MODERATE AGREEMENT: Review all DISPUTED points carefully. "
            "Keep only those that clearly appear verbatim (or very close) in the original text."
        )
    else:
        lines.append(
            "LOW AGREEMENT: Models disagree significantly — likely an OCR quality issue. "
            "Be very strict: only include points that are verbatim in the original text."
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick stats helper (used in logging / response metadata)
# ---------------------------------------------------------------------------

def comparison_stats(comparison: ComparisonResult) -> dict:
    """Return a compact stats dict suitable for JSON response metadata."""
    return {
        "agreement_rate":   comparison["agreement_rate"],
        "agreed_positive":  len(comparison["agreed_positive"]),
        "agreed_negative":  len(comparison["agreed_negative"]),
        "disputed_positive": len(comparison["disputed_positive"]),
        "disputed_negative": len(comparison["disputed_negative"]),
        "total_points":     comparison["total_points"],
    }
