"""Score sentences or a whole article from the command line.

    python predict.py --model outputs/deberta-babe/best "Lawmakers rammed through the reckless bill."
    python predict.py --model outputs/deberta-babe/best --file article.txt
"""
from __future__ import annotations

import argparse
import json

from src.scorer import BiasScorer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--file", help="Text file to score as one article")
    p.add_argument("sentences", nargs="*")
    args = p.parse_args()

    scorer = BiasScorer(args.model)
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            print(json.dumps(scorer.score_article(f.read()), indent=2, ensure_ascii=False))
    for s in scorer.score_sentences(args.sentences):
        print(f"[{s.label:>8}] {s.probs}  {s.text}")


if __name__ == "__main__":
    main()
