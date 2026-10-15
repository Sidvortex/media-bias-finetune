"""Fine-tune an encoder for media-bias classification.

    python train.py --config configs/deberta_babe.yaml
    python train.py --config configs/xlmr_babe.yaml
    python train.py --config configs/deberta_babe.yaml --set data.label_column=type \
        "data.labels=[left,center,right]" train.output_dir=outputs/deberta-lean
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import base_parser, load_config
from src.data import class_counts, load_splits


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy (for imbalanced labels)."""

    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main() -> None:
    args = base_parser(__doc__, "configs/deberta_babe.yaml").parse_args()
    cfg = load_config(args.config, args.set)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    labels = dcfg["labels"]
    out = Path(tcfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)

    ds = load_splits(dcfg)
    counts = class_counts(ds["train"], len(labels))
    print({k: len(v) for k, v in ds.items()}, "train class counts:", dict(zip(labels, counts)))

    tok = AutoTokenizer.from_pretrained(mcfg["name"])
    ds = ds.map(
        lambda b: tok(b["text"], truncation=True, max_length=dcfg["max_length"]),
        batched=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        mcfg["name"],
        num_labels=len(labels),
        id2label=dict(enumerate(labels)),
        label2id={l: i for i, l in enumerate(labels)},
        ignore_mismatched_sizes=True,  # lets you start from a checkpoint with a different head
    )

    weights = None
    if tcfg["class_weights"]:
        c = torch.tensor(counts, dtype=torch.float)
        weights = c.sum() / (len(c) * c.clamp(min=1))

    targs = TrainingArguments(
        output_dir=str(out / "checkpoints"),
        num_train_epochs=tcfg["epochs"],
        learning_rate=tcfg["lr"],
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg["batch_size"] * 2,
        weight_decay=tcfg["weight_decay"],
        warmup_steps=tcfg["warmup_ratio"],  # float < 1 = ratio of total steps
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=25,
        report_to="none",
        seed=dcfg["seed"],
    )

    trainer = WeightedTrainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tok,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=metrics_fn,
        callbacks=[EarlyStoppingCallback(tcfg["early_stopping_patience"])],
        class_weights=weights,
    )
    trainer.train()

    best = out / "best"
    trainer.save_model(str(best))
    tok.save_pretrained(str(best))

    pred = trainer.predict(ds["test"])
    y_pred = pred.predictions.argmax(-1)
    y_true = pred.label_ids
    report = classification_report(y_true, y_pred, target_names=labels, digits=4, output_dict=True)
    results = {
        "model": mcfg["name"],
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro"),
        "per_class": {l: report[l] for l in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "labels": labels,
    }
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print(f"Best model saved to {best}")


if __name__ == "__main__":
    main()
