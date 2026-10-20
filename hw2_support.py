"""Provided, ungraded utilities for CSCI 544 HW2. Do not submit/modify this file.
No data access, training, printing, or downloads occur when imported.
"""
from __future__ import annotations
import copy
import csv
import html
import random
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split


def tokenize_review(text: str) -> list[str]:
    """HW2's intentionally minimal tokenizer; this is not HW1 preprocessing."""
    text = html.unescape(text).lower()
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def load_reviews(path: str | Path, n_per_class: int = 5000):
    """Return train, validation, test lists of (token_list, binary_label).

    This repeats HW1's TSV parsing / validity rules, but uses a NEW smaller split.
    Sampling and splitting precede tokenization and vocabulary fitting.
    """
    if n_per_class < 10 or n_per_class % 10:
        raise ValueError("n_per_class must be a positive multiple of 10")
    df = pd.read_csv(path, sep="\t", dtype=str, quoting=csv.QUOTE_NONE,
                     keep_default_na=False, on_bad_lines="skip",
                     usecols=["review_body", "star_rating"])
    df = df.rename(columns={"review_body": "review", "star_rating": "rating"})
    ratings = pd.to_numeric(df["rating"], errors="coerce")
    valid = df["review"].str.strip().ne("") & ratings.isin([1, 2, 3, 4, 5])
    df = df.loc[valid].copy()
    df["rating"] = ratings.loc[valid].astype(int)
    pos = df.loc[df["rating"].isin([4, 5])].copy()
    neg = df.loc[df["rating"].isin([1, 2])].copy()
    if min(len(pos), len(neg)) < n_per_class:
        raise ValueError("Not enough valid reviews in each class; do not substitute data")
    pos["label"], neg["label"] = 1, 0
    sample = pd.concat([pos.sample(n=n_per_class, random_state=42),
                        neg.sample(n=n_per_class, random_state=42)], ignore_index=True)
    trainval, test = train_test_split(sample, test_size=0.1,
                                     stratify=sample["label"], random_state=42)
    train, val = train_test_split(trainval, test_size=1 / 9,
                                  stratify=trainval["label"], random_state=42)
    def convert(frame):
        return [(tokenize_review(text), int(y))
                for text, y in zip(frame["review"], frame["label"])]
    return convert(train), convert(val), convert(test)


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(examples, collate_fn, batch_size=64, shuffle=False, seed=42):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(examples, batch_size=batch_size, shuffle=shuffle,
                      generator=generator, collate_fn=collate_fn,
                      num_workers=0, drop_last=False)


def make_optimizer(parameters, lr=1e-3):
    # Explicit settings keep this assignment independent of future defaults.
    return torch.optim.Adam(parameters, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=0, amsgrad=False, foreach=False)


def fit_classifier(model, train_loader, val_loader, train_fn, eval_fn,
                   epochs=5, lr=1e-3):
    """Provided checkpoint manager: select by validation loss, never test data."""
    optimizer = make_optimizer(model.parameters(), lr)
    best_state, best_loss, best_epoch, best_metrics = None, float("inf"), 0, None
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_fn(model, train_loader, optimizer)
        metrics = eval_fn(model, val_loader)
        history.append({"epoch": epoch, "train_loss": train_loss,
                        "validation_loss": metrics["loss"]})
        if metrics["loss"] < best_loss:  # strict: retain earlier epoch on a tie
            best_state = copy.deepcopy(model.state_dict())
            best_loss, best_epoch = metrics["loss"], epoch
            best_metrics = dict(metrics)
    model.load_state_dict(best_state)
    return {"best_epoch": best_epoch, "validation": best_metrics, "history": history}


def pretrain_mlm(encoder, lm_head, train_loader, corrupt_fn, loss_fn,
                 epochs=2, lr=1e-3, seed=43):
    """Provided pretraining harness. Sentiment labels are deliberately ignored.

    Project ONLY selected hidden states into the vocabulary, saving memory and
    compute. Loss is still exactly the selected-position MLM objective.
    """
    optimizer = make_optimizer(list(encoder.parameters()) + list(lm_head.parameters()), lr)
    rng = torch.Generator().manual_seed(seed)
    vocab_size = lm_head.out_features
    history = []
    for _ in range(epochs):
        encoder.train()
        lm_head.train()
        total_loss, total_targets = 0.0, 0
        for ids, mask, _lengths, _unused_labels in train_loader:
            select_u = torch.rand(ids.shape, generator=rng)
            replace_u = torch.rand(ids.shape, generator=rng)
            random_ids = (torch.randint(3, vocab_size, ids.shape, generator=rng)
                          if vocab_size > 3 else torch.zeros_like(ids))
            corrupted, targets = corrupt_fn(ids, select_u, replace_u, random_ids)
            selected = targets.ne(-100)
            count = int(selected.sum().item())
            if count == 0:  # no optimizer step, even if Adam has momentum state
                continue
            optimizer.zero_grad(set_to_none=True)
            hidden = encoder(corrupted, mask)
            logits = lm_head(hidden[selected])
            loss = loss_fn(logits, targets[selected])
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item()) * count
            total_targets += count
        history.append({"loss": total_loss / total_targets if total_targets else 0.0,
                        "n_targets": total_targets})
    return history
