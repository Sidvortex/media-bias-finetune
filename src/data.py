"""Load BABE (or your own CSV) into train / val / test with integer labels."""
from __future__ import annotations

from datasets import ClassLabel, Dataset, DatasetDict, load_dataset


def _normalise_labels(ds: Dataset, label_col: str, labels: list[str]) -> Dataset:
    """Map string or int labels to 0..n-1 and drop rows with unknown/missing labels."""
    lookup = {name.lower(): i for i, name in enumerate(labels)}

    def convert(ex):
        v = ex[label_col]
        if v is None:
            return {"labels": -1}
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            iv = int(v)
            return {"labels": iv if 0 <= iv < len(labels) else -1}
        key = str(v).strip().lower()
        if key.lstrip("-").isdigit():
            iv = int(key)
            return {"labels": iv if 0 <= iv < len(labels) else -1}
        return {"labels": lookup.get(key, -1)}

    ds = ds.map(convert)
    ds = ds.filter(lambda ex: ex["labels"] >= 0)
    return ds.cast_column("labels", ClassLabel(names=labels))


def load_splits(cfg: dict) -> DatasetDict:
    text_col, label_col, labels = cfg["text_column"], cfg["label_column"], cfg["labels"]
    if cfg["source"] == "hf":
        raw = load_dataset(cfg["dataset"])
        train, test = raw["train"], raw["test"]
    elif cfg["source"] == "csv":
        raw = load_dataset("csv", data_files={"train": cfg["train_csv"], "test": cfg["test_csv"]})
        train, test = raw["train"], raw["test"]
    else:
        raise ValueError(f"Unknown data.source: {cfg['source']}")

    def prep(ds: Dataset) -> Dataset:
        ds = ds.filter(lambda ex: isinstance(ex[text_col], str) and ex[text_col].strip() != "")
        ds = _normalise_labels(ds, label_col, labels)
        keep = {text_col, "labels"}
        ds = ds.remove_columns([c for c in ds.column_names if c not in keep])
        if text_col != "text":
            ds = ds.rename_column(text_col, "text")
        return ds

    train, test = prep(train), prep(test)
    split = train.train_test_split(
        test_size=cfg["val_fraction"], seed=cfg["seed"], stratify_by_column="labels"
    )
    return DatasetDict(train=split["train"], validation=split["test"], test=test)


def class_counts(ds: Dataset, n: int) -> list[int]:
    counts = [0] * n
    for y in ds["labels"]:
        counts[y] += 1
    return counts
