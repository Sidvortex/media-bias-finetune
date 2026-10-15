"""Article-level bias scoring on top of the sentence classifier."""
from __future__ import annotations

import re
from dataclasses import dataclass

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Splits on . ! ? and the Devanagari danda, keeps reasonably long sentences.
_SENT_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def split_sentences(text: str, min_chars: int = 25) -> list[str]:
    parts = [s.strip() for s in _SENT_SPLIT.split(text.replace("\n", " "))]
    return [s for s in parts if len(s) >= min_chars]


@dataclass
class SentenceScore:
    text: str
    label: str
    probs: dict[str, float]


class BiasScorer:
    def __init__(self, model_dir: str, device: str | None = None, max_length: int = 160):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.labels = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]
        self.max_length = max_length

    @torch.no_grad()
    def score_sentences(self, sentences: list[str], batch_size: int = 32) -> list[SentenceScore]:
        out: list[SentenceScore] = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i : i + batch_size]
            enc = self.tok(
                batch, truncation=True, max_length=self.max_length, padding=True, return_tensors="pt"
            ).to(self.device)
            probs = self.model(**enc).logits.float().softmax(-1).cpu()
            for s, p in zip(batch, probs):
                pd = {l: round(float(v), 4) for l, v in zip(self.labels, p)}
                out.append(SentenceScore(s, self.labels[int(p.argmax())], pd))
        return out

    def score_article(self, text: str, biased_label: str = "biased", top_k: int = 5) -> dict:
        """Returns a 0-100 framing score plus the most loaded sentences.

        framing_score = mean P(biased) over sentences x 100. This maps directly onto the
        "framing intensity" axis of a multi-axis bias tool.
        """
        sents = split_sentences(text) or [text.strip()]
        scored = self.score_sentences(sents)
        if biased_label not in self.labels:
            return {"sentences": [s.__dict__ for s in scored], "labels": self.labels}
        pb = [s.probs[biased_label] for s in scored]
        ranked = sorted(scored, key=lambda s: s.probs[biased_label], reverse=True)
        return {
            "framing_score": round(100 * sum(pb) / len(pb), 1),
            "biased_fraction": round(sum(p >= 0.5 for p in pb) / len(pb), 3),
            "n_sentences": len(pb),
            "top_biased": [{"text": s.text, "p_biased": s.probs[biased_label]} for s in ranked[:top_k]],
        }
