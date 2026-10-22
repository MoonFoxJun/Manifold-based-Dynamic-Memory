"""Basic tests for the recall pipeline (synthetic archive, no data needed)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from semantic_memory import build_index, select, grams  # noqa: E402

DAY = 86400e3


def _archive():
    now = 1_700_000_000_000
    texts = [
        "the kernel trick turns inner products into similarities",
        "leave one out cross validation uses the hat matrix diagonal",
        "核函数是把内积变成相似度的技巧",
        "留一法用帽矩阵对角线计算误差",
        "留一法的误差等于残差除以一减杠杆值",
    ]
    return [{"seq": i, "role": "user", "text": t, "time": now - (len(texts) - i) * DAY}
            for i, t in enumerate(texts)]


def test_grams_tokenize():
    g = grams("核函数 kernel")
    assert "c:核函" in g and "w:kernel" in g


def test_build_index_normalized():
    msgs = _archive()
    X, idx, idf = build_index(msgs, min_df=1)
    assert X.shape[0] == len(msgs)
    norms = np.linalg.norm(X, axis=1)
    # every document has at least one in-vocabulary token with min_df=1
    assert np.allclose(norms, 1.0)


def test_select_returns_relevant_messages():
    msgs = _archive()
    X, idx, idf = build_index(msgs, min_df=1)
    blocks, stats = select("留一法 杠杆", msgs, X, idx, idf, {}, budget=2000,
                           recent=0.0, gamma=0.0, fresh_penalty=0.0,
                           fresh_days=1e9, update=False)
    assert blocks, "expected at least one result"
    returned = {r for r, _ in blocks}
    assert returned & {3, 4}
    assert stats["verdict"] in ("solid", "weak")


def test_refusal_for_unknown_topic():
    msgs = _archive()
    X, idx, idf = build_index(msgs, min_df=1)
    blocks, stats = select("zzzqqq", msgs, X, idx, idf, {}, budget=2000, update=False)
    assert blocks == []
    assert "no term" in stats["reason"] or "too common" in stats["reason"]


def test_dedup_folds_near_duplicates():
    now = 1_700_000_000_000
    msgs = [
        {"seq": 0, "role": "user", "text": "留一法误差等于残差除以一减杠杆值", "time": now},
        {"seq": 1, "role": "user", "text": "留一法误差等于残差除以一减杠杆值。", "time": now - DAY},
        {"seq": 2, "role": "user", "text": "流形学习需要图拉普拉斯", "time": now - 2 * DAY},
    ]
    X, idx, idf = build_index(msgs, min_df=1)
    blocks, stats = select("留一法 杠杆", msgs, X, idx, idf, {}, budget=2000,
                           recent=0.0, gamma=0.0, fresh_penalty=0.0,
                           fresh_days=1e9, update=False)
    assert blocks, stats
    assert stats["clusters"] <= stats["pool"]
    assert stats["folded"] >= 1 or stats["clusters"] < stats["pool"]


def test_budget_is_respected():
    msgs = _archive()
    X, idx, idf = build_index(msgs, min_df=1)
    blocks, stats = select("留一法", msgs, X, idx, idf, {}, budget=80,
                           recent=0.0, gamma=0.0, fresh_penalty=0.0,
                           fresh_days=1e9, update=False)
    if blocks:
        total = sum(len(msgs[r]["text"]) for r, _ in blocks)
        assert total <= 80 + 30 * stats["folded"] + 1
