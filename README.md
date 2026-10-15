# media-bias-finetune

Fine-tunes transformer encoders to detect **biased / loaded language in news sentences**, then scores whole articles. It's meant to replace keyword-based scoring in a multi-axis media-bias tool. The article score maps onto a "framing intensity" axis, and the FastAPI service drops in next to an existing backend.

## Models and data

| Config | Encoder | Why |
|---|---|---|
| `deberta_babe.yaml` | DeBERTa-v3-base | Strongest English baseline |
| `xlmr_babe.yaml` | XLM-RoBERTa-base | Multilingual, so it can continue onto Hindi / Indian-English news |
| `indian_news_csv.yaml` | starts from the XLM-R run | Second-stage fine-tune on your own labelled CSV |

Training data: [BABE](https://huggingface.co/datasets/mediabiasgroup/BABE) (Bias Annotations By Experts): about 3.1k train and 1k test sentences labelled by trained annotators. The original paper reports about 0.80 macro-F1, which is a sensible target.

`label` is used as `0 = neutral, 1 = biased`. Check a few rows on the dataset page to confirm the direction before reporting results.

BABE also has a `type` column (the outlet's left / center / right lean). To train a lean classifier:

```bash
python train.py --set data.label_column=type "data.labels=[left,center,right]" train.output_dir=outputs/deberta-lean
```

## Run on Kaggle

Push to `github.com/Sidvortex/media-bias-finetune`, import `kaggle_notebook.ipynb`, enable GPU and Internet, and run all. Each run takes about 10 minutes on a T4.

## Use it locally (CPU is fine for inference)

```bash
pip install --break-system-packages -r requirements.txt
python predict.py --model outputs/deberta-babe/best "Lawmakers rammed through the reckless bill."
python predict.py --model outputs/deberta-babe/best --file article.txt

MODEL_DIR=outputs/deberta-babe/best uvicorn serve:app --port 8010
curl -X POST localhost:8010/score/article -H 'content-type: application/json' -d '{"text": "..."}'
```

`/score/article` returns:

```json
{"framing_score": 41.3, "biased_fraction": 0.375, "n_sentences": 8,
 "top_biased": [{"text": "...", "p_biased": 0.94}]}
```

## Adapting to Indian news (the non-Western gap)

BABE is US news. To make the model reliable on Indian outlets:

1. Collect 500 to 1,500 sentences from Indian articles across outlets and topics.
2. Label them `neutral` / `biased` (two annotators plus a tie-break if you can; report agreement).
3. Save them as `data/train.csv` and `data/test.csv` with columns `text,label`.
4. Run `python train.py --config configs/indian_news_csv.yaml`.

Report BABE-only vs BABE+Indian on your Indian test set. That before/after is the interesting result.

## Outputs

`outputs/<run>/best/` (model and tokenizer) and `outputs/<run>/results.json` (accuracy, macro-F1, per-class precision/recall, confusion matrix).
