"""FastAPI service so the media-bias tool's backend can call the fine-tuned model.

    MODEL_DIR=outputs/deberta-babe/best uvicorn serve:app --port 8010

    POST /score/article   {"text": "..."}          -> framing_score, top_biased sentences
    POST /score/sentences {"sentences": ["..."]}   -> per-sentence labels + probabilities
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel

from src.scorer import BiasScorer

app = FastAPI(title="Media bias classifier")
scorer = BiasScorer(os.environ.get("MODEL_DIR", "outputs/deberta-babe/best"))


class Article(BaseModel):
    text: str
    top_k: int = 5


class Sentences(BaseModel):
    sentences: list[str]


@app.get("/health")
def health() -> dict:
    return {"ok": True, "labels": scorer.labels, "device": scorer.device}


@app.post("/score/article")
def score_article(body: Article) -> dict:
    return scorer.score_article(body.text, top_k=body.top_k)


@app.post("/score/sentences")
def score_sentences(body: Sentences) -> list[dict]:
    return [s.__dict__ for s in scorer.score_sentences(body.sentences)]
