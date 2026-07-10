"""
sentiment.py — Sentence splitting and multilingual sentiment classification.

Model: cardiffnlp/twitter-xlm-roberta-base-sentiment
  - Supports English, French, Arabic (and many other languages)
  - Labels: positive, negative, neutral

Sentence splitting:
  - Primary: spaCy sentencizer (language-agnostic pipeline)
  - Fallback: nltk sent_tokenize
"""

import re
import logging
from typing import TypedDict

from transformers import pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentiment model (loaded once at module import time)
# ---------------------------------------------------------------------------

SENTIMENT_MODEL = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

logger.info(f"Loading sentiment model: {SENTIMENT_MODEL}")
_sentiment_pipe = pipeline(
    "sentiment-analysis",
    model=SENTIMENT_MODEL,
    tokenizer=SENTIMENT_MODEL,
    top_k=1,           # return only the best label
    truncation=True,   # handle sentences longer than 512 tokens
    max_length=512,
)
logger.info("Sentiment model loaded successfully.")

# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

# Try spaCy first (best for multilingual tokenisation)
try:
    import spacy
    # We only need the sentencizer; load a blank multilingual model to avoid
    # downloading large language-specific models.
    _nlp = spacy.blank("xx")          # "xx" = multilingual blank model
    _nlp.add_pipe("sentencizer")
    SPACY_AVAILABLE = True
    logger.info("spaCy sentencizer ready.")
except Exception as exc:
    SPACY_AVAILABLE = False
    logger.warning(f"spaCy not available: {exc}. Will fall back to nltk.")

# NLTK fallback
try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import sent_tokenize
    NLTK_AVAILABLE = True
except Exception as exc:
    NLTK_AVAILABLE = False
    logger.warning(f"nltk not available: {exc}.")


def split_sentences(text: str) -> list[str]:
    """
    Split `text` into a list of individual sentences.

    Uses spaCy sentencizer when available, otherwise falls back to
    nltk sent_tokenize, and finally to a simple regex splitter.
    """
    if not text.strip():
        return []

    if SPACY_AVAILABLE:
        doc = _nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        return sentences

    if NLTK_AVAILABLE:
        # nltk works best with English; it degrades gracefully for others
        sentences = sent_tokenize(text)
        return [s.strip() for s in sentences if s.strip()]

    # Last-resort regex split on sentence-ending punctuation
    logger.warning("Using regex-based sentence splitter (may be inaccurate).")
    parts = re.split(r"(?<=[.!?؟])\s+", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Sentiment analysis
# ---------------------------------------------------------------------------

# Map the model's raw labels to our canonical label set
_LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral":  "neutral",
    # Some versions of the model use title-case
    "Positive": "positive",
    "Negative": "negative",
    "Neutral":  "neutral",
}

# Minimum sentence length to bother classifying (very short fragments are noisy)
_MIN_SENTENCE_CHARS = 10


class SentimentResult(TypedDict):
    sentence: str
    label: str      # "positive" | "negative" | "neutral"
    score: float    # confidence score 0–1


def classify_sentences(sentences: list[str]) -> list[SentimentResult]:
    """
    Run sentiment classification on each sentence.

    Returns a list of SentimentResult dicts, one per input sentence.
    Very short sentences (< _MIN_SENTENCE_CHARS characters) are labelled
    'neutral' without running the model to save compute.
    """
    results: list[SentimentResult] = []

    for sentence in sentences:
        if not sentence.strip():
            continue

        # Skip fragments that are too short to carry meaningful sentiment
        if len(sentence.strip()) < _MIN_SENTENCE_CHARS:
            results.append({
                "sentence": sentence,
                "label": "neutral",
                "score": 1.0,
            })
            continue

        try:
            # pipeline returns [[{"label": ..., "score": ...}]]
            output = _sentiment_pipe(sentence)
            best = output[0][0] if isinstance(output[0], list) else output[0]
            raw_label = best["label"]
            label = _LABEL_MAP.get(raw_label, "neutral")
            results.append({
                "sentence": sentence,
                "label": label,
                "score": round(float(best["score"]), 4),
            })
        except Exception as exc:
            logger.warning(f"Sentiment model failed on sentence: {exc}")
            results.append({
                "sentence": sentence,
                "label": "neutral",
                "score": 0.0,
            })

    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class AnalysisOutput(TypedDict):
    raw_text: str
    positive: list[str]
    negative: list[str]
    neutral: list[str]


def analyse_text(raw_text: str) -> AnalysisOutput:
    """
    Split `raw_text` into sentences, classify each, and bucket them.

    Returns an AnalysisOutput with:
      - raw_text: the original input
      - positive: sentences labelled positive
      - negative: sentences labelled negative
      - neutral: sentences labelled neutral
    """
    sentences = split_sentences(raw_text)
    classified = classify_sentences(sentences)

    positive = [r["sentence"] for r in classified if r["label"] == "positive"]
    negative = [r["sentence"] for r in classified if r["label"] == "negative"]
    neutral  = [r["sentence"] for r in classified if r["label"] == "neutral"]

    return AnalysisOutput(
        raw_text=raw_text,
        positive=positive,
        negative=negative,
        neutral=neutral,
    )
