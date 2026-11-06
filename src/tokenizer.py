"""Field-marked word tokenizer and train-only vocabulary.

Sequence layout:  [CLS] [ILCE] ilce_<district> [YIL] yil_<year> [MEKAN] <name tokens>

* district / year tokens carry a prefix (ilce_, yil_) so that the word 'silivri'
  in "Silivri Kapı Buz Pisti" (a rink in Fatih) is not the district token;
* the vocabulary is built ONLY from the training part it is used with;
* min_freq = 2: tokens seen once in training map to [UNK], so the [UNK] embedding
  is actually trained (otherwise unseen test tokens would hit a random vector).
"""
from __future__ import annotations

from collections import Counter

import torch

from common import tokenize_tr

SPECIALS = ["[PAD]", "[UNK]", "[CLS]", "[ILCE]", "[YIL]", "[MEKAN]"]
PAD, UNK, CLS = 0, 1, 2


def record_tokens(district: str, year: int, name: str, use_year: bool = True) -> list[str]:
    toks = ["[CLS]", "[ILCE]"] + ["ilce_" + t for t in tokenize_tr(district)]
    if use_year:
        toks += ["[YIL]", f"yil_{int(year)}"]
    toks += ["[MEKAN]"] + tokenize_tr(name)
    return toks


class Vocab:
    def __init__(self, min_freq: int = 2):
        self.min_freq = min_freq
        self.itos: list[str] = list(SPECIALS)
        self.stoi: dict[str, int] = {t: i for i, t in enumerate(self.itos)}
        self.counts: Counter = Counter()

    def fit(self, token_lists):
        self.counts = Counter(t for toks in token_lists for t in toks if t not in SPECIALS)
        for tok, c in sorted(self.counts.items()):
            if c >= self.min_freq and tok not in self.stoi:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
        return self

    def __len__(self):
        return len(self.itos)

    def encode(self, toks):
        return [self.stoi.get(t, UNK) for t in toks]


def batch_encode(vocab: Vocab, token_lists, max_len: int | None = None):
    """Returns ids [B, T] (right padded with [PAD]) and n_truncated."""
    seqs = [vocab.encode(t) for t in token_lists]
    T = max(len(s) for s in seqs) if max_len is None else max_len
    n_trunc = sum(len(s) > T for s in seqs)
    ids = torch.full((len(seqs), T), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = s[:T]
        ids[i, : len(s)] = torch.tensor(s)
    return ids, n_trunc
