"""Objective evaluation set for the recall pipeline.

Each query is paired with keywords that mark a message as relevant (gold set =
messages whose text contains any keyword).  This is not a perfect benchmark;
it is a fixed number to compare against when the pipeline changes.

Run from the project root::

    python -m eval.eval_set
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_memory import build_index, select, load_messages  # noqa: E402

EVAL = [
    ("loocv 留一法", ["留一", "loocv"]),
    ("核函数", ["核函数", "核函"]),
    ("流形", ["流形"]),
    ("对偶", ["对偶"]),
    ("压缩", ["压缩"]),
    ("张量", ["张量"]),
    ("扩散映射", ["扩散映射", "扩散"]),
    ("归一化", ["归一化"]),
    ("条件数", ["条件数"]),
    ("三明治", ["三明治"]),
]


def main(budget=4000, k=5):
    msgs = load_messages()
    X, idx, idf = build_index(msgs)

    print(f"{'query':<20}{'gold':>6}{'got':>5}{'hit':>5}{'P@k':>6}{'R@k':>6}")
    tP = tR = n = 0
    for q, kws in EVAL:
        gold = {i for i, m in enumerate(msgs) if any(w in m["text"] for w in kws)}
        blocks, stats = select(q, msgs, X, idx, idf, {}, budget=budget, pool=60,
                               dedup=True, dedup_thr=-1, recent=0.0, gamma=0.0,
                               fresh_penalty=0.0, fresh_days=1e9, update=False)
        if not blocks:
            print(f"{q:<20}{len(gold):>6}{0:>5}{0:>5}{0:>6}{0:>6}  ({stats.get('reason', '')})")
            continue
        got = [r for r, _ in blocks][:k]
        hit = sum(1 for i in got if i in gold)
        P = hit / max(1, len(got))
        R = hit / max(1, min(k, len(gold)))
        tP += P
        tR += R
        n += 1
        print(f"{q:<20}{len(gold):>6}{len(got):>5}{hit:>5}{P:>6.2f}{R:>6.2f}")

    print(f"\nmean P@{k} = {tP / max(n, 1):.3f}   R@{k} = {tR / max(n, 1):.3f}")


if __name__ == "__main__":
    main()
