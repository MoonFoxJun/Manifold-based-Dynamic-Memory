"""Recall pipeline: relevance, adaptive suppression, dedup, budgeted selection.

The pipeline is intentionally extractive: it never generates text.  It turns a
keyword query into a small, ordered, de-duplicated passage with provenance
pointers back to the archive.

Stages
------
1. relevance      cosine between the query vector and every message
2. modifiers      recency / freshness / access-strength weights
3. verdict        solid / weak / refuse (honest "no relevant memory")
4. crowding       how many strong hits compete for the character budget
5. dedup          content clustering, one representative per cluster
6. selection      MMR under the budget, the rest folded into pointers
"""

import datetime
import json
import math
import os
import time

import numpy as np

from . import features

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("SM_DATA", os.path.join(PROJECT_ROOT, "data", "messages.jsonl"))
STATE = os.environ.get("SM_STATE", os.path.join(PROJECT_ROOT, ".state", "memory_state.json"))


# ---------- persistent memory state ----------

def load_state():
    """Load the per-message access state (hit counts, last-returned time)."""
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state):
    os.makedirs(os.path.dirname(STATE), exist_ok=True)
    with open(STATE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=1)


# ---------- scoring ----------

def score(query, msgs, X, idx, idf, state,
          recent=0.5, gamma=0.6, tau_days=21.0,
          fresh_penalty=0.6, fresh_days=7.0, max_sent=True,
          time_center=None, time_sigma_days=10.0, time_hard=False):
    """Return ``(base_scores, relevance)`` per message.

    Long queries are split into sentences and each message keeps its best
    sentence score, which stops a paragraph from diluting the query vector.

    ``time_center`` (epoch ms) switches the recency factor to a Gaussian time
    window centred there; that is what the "dig deeper" action slides backwards.
    """
    sents = [s.strip() for s in __import__("re").split(r"[。！？!?；;\n]+|\.\s", query) if s.strip()]
    if not max_sent or len(sents) <= 1:
        sents = [query]
    vecs = [features.query_vector(s, idx, idf) for s in sents]
    vecs = [v for v in vecs if v.any()]
    if not vecs:
        return None, None
    rel = np.max(np.stack([X @ v for v in vecs], axis=0), axis=0)

    t = np.array([m["time"] for m in msgs], dtype=float)
    t_max = t.max()
    if time_center is None:
        rec = np.exp(-(t_max - t) / (tau_days * 86400e3))
    else:
        rec = np.exp(-0.5 * ((t - time_center) / (time_sigma_days * 86400e3)) ** 2)

    now = time.time() * 1000
    hits = np.zeros(len(msgs))
    fresh = np.zeros(len(msgs))
    for i, m in enumerate(msgs):
        e = state.get(str(m["seq"]), {})
        hits[i] = e.get("hits", 0)
        last = e.get("last", 0)
        if last:
            fresh[i] = math.exp(-(now - last) / (fresh_days * 86400e3))

    strength = 1.0 + gamma * np.log1p(hits)
    suppress = 1.0 - fresh_penalty * fresh
    base = rel * (rec ** recent) * strength * np.maximum(suppress, 0.15)

    if time_center is not None and time_hard:
        base = base * (np.abs(t - time_center) <= 3 * time_sigma_days * 86400e3)
    return base, rel


# ---------- selection ----------

def select(query, msgs, X, idx, idf, state,
           budget=3000, pool=60, lam=0.55, dedup=True, dedup_thr=-1,
           recent=0.5, gamma=0.6, tau_days=21.0,
           fresh_penalty=0.6, fresh_days=7.0, update=True, max_sent=True,
           time_center=None, time_sigma_days=10.0, time_hard=False,
           exclude=None, exclude_hard=False, min_score=0.15):
    """Run the full recall pipeline.

    Returns ``(blocks, stats)`` where ``blocks`` is a list of
    ``(representative_index, folded_indices)``.
    """
    base, rel = score(query, msgs, X, idx, idf, state,
                      recent=recent, gamma=gamma, tau_days=tau_days,
                      fresh_penalty=fresh_penalty, fresh_days=fresh_days,
                      max_sent=max_sent, time_center=time_center,
                      time_sigma_days=time_sigma_days, time_hard=time_hard)
    if base is None:
        any_in_vocab = any(w in idx for w in features.grams(query))
        reason = ("query too common / too broad, try more specific terms"
                  if any_in_vocab else "query has no term in the memory vocabulary")
        return [], {"reason": reason}

    # honest refusal: decide on *pure* relevance, not on time-weighted scores
    top_score = float(rel.max())
    if top_score <= 0:
        return [], {"reason": "no relevant memory"}
    strong = int((rel > 0.5 * top_score).sum())
    if top_score < min_score:
        return [], {"reason": f"relevance too low (top {top_score:.3f} < {min_score}); "
                              f"no reliable memory"}
    verdict = "solid" if top_score >= 0.30 else "weak"

    if exclude:
        for i in exclude:
            if 0 <= i < len(base):
                base[i] = 0.0 if exclude_hard else base[i] * 0.1

    order = [int(i) for i in np.argsort(-base) if base[i] > 0][:pool]

    # crowding-adaptive suppression: many strong hits per unit of budget means
    # we must compress harder; few hits means we can afford to keep more.
    n_match = int((base > 0.25 * base.max()).sum()) if base.max() > 0 else 0
    lens = [len(msgs[i]["text"]) for i in order] or [1]
    capacity = max(1.0, budget / max(1.0, sum(lens) / len(lens)))
    crowding = n_match / capacity
    lam_eff = lam * min(4.0, max(0.4, crowding))

    # adaptive dedup threshold: long messages dilute cosine, so a fixed
    # threshold cannot work; use the pool's own similarity distribution.
    if dedup and dedup_thr is not None and dedup_thr < 0:
        if len(order) > 1:
            S = X[order] @ X[order].T
            iu = np.triu_indices(len(order), 1)
            dedup_thr = float(np.clip(S[iu].mean() + S[iu].std(), 0.10, 0.40))
        else:
            dedup_thr = 0.20

    reps, folded = [], {}
    for i in order:
        if dedup and reps:
            sims = [float(X[i] @ X[r]) for r in reps]
            best = int(np.argmax(sims))
            if sims[best] > dedup_thr:
                folded[reps[best]].append(i)
                continue
        reps.append(i)
        folded[i] = []

    selected, used = [], 0
    for i in sorted(reps, key=lambda j: -base[j]):
        size = len(msgs[i]["text"]) + 30 * max(1, len(folded[i]))
        if used + size > budget:
            continue
        if selected:
            sim = max(float(X[i] @ X[j]) for j in selected)
            if base[i] - lam_eff * sim * base.max() <= 0:
                continue
        selected.append(i)
        used += size
        if used >= budget:
            break

    if update:
        now = int(time.time() * 1000)
        for r in selected:
            for j in [r] + folded[r]:
                entry = state.setdefault(str(msgs[j]["seq"]), {"hits": 0})
                entry["hits"] += 1
                entry["last"] = now
        save_state(state)

    selected.sort()
    stats = {
        "pool": len(order),
        "matched": n_match,
        "verdict": verdict,
        "top_score": top_score,
        "strong": strong,
        "crowding": crowding,
        "lam_eff": lam_eff,
        "clusters": len(reps),
        "reps": len(selected),
        "folded": sum(len(folded[r]) for r in selected),
        "chars": used,
    }
    return [(r, folded[r]) for r in selected], stats


# ---------- presentation ----------

def fmt_time(ms):
    return datetime.datetime.fromtimestamp(ms / 1000).strftime("%m-%d %H:%M")


def print_result(blocks, stats, msgs, budget):
    print(f"=== budget {budget} chars | verdict {stats.get('verdict', '?')} "
          f"(top {stats.get('top_score', 0):.3f}, strong {stats.get('strong', 0)}) | "
          f"matched {stats.get('matched', 0)}, crowding {stats.get('crowding', 0):.1f}, "
          f"suppression {stats.get('lam_eff', 0):.2f} | pool {stats.get('pool', 0)} -> "
          f"{stats.get('clusters', 0)} clusters -> {stats.get('reps', 0)} reps "
          f"({stats.get('folded', 0)} folded) | ~{stats.get('chars', 0)} chars ===")
    for r, fo in blocks:
        m = msgs[r]
        print("\n" + "-" * 60)
        print(f"[#{m['seq']:>4} {fmt_time(m['time'])} {m['role']}]")
        print(m["text"].strip()[:600])
        if fo:
            seqs = " ".join(f"#{msgs[j]['seq']}" for j in sorted(fo)[:12])
            more = " ..." if len(fo) > 12 else ""
            print(f"   +- {len(fo)} similar folded (drill down): {seqs}{more}")


def load_messages(path=None):
    path = path or DATA
    # utf-8-sig tolerates a byte-order mark (common when files are edited on Windows)
    return [json.loads(line) for line in open(path, encoding="utf-8-sig") if line.strip()]


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Extractive recall over a message archive.")
    ap.add_argument("query")
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--recent", type=float, default=0.5, help="recency exponent, 0 = ignore time")
    ap.add_argument("--lam", type=float, default=0.55, help="MMR diversity strength")
    ap.add_argument("--gamma", type=float, default=0.6, help="access-strength gain")
    ap.add_argument("--pool", type=int, default=60)
    ap.add_argument("--dedup-thr", type=float, default=-1, help="negative = adaptive")
    ap.add_argument("--fresh", type=float, default=0.6, help="freshness suppression 0..1")
    ap.add_argument("--fresh-days", type=float, default=7.0)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--no-update", action="store_true")
    ap.add_argument("--data", default=DATA)
    args = ap.parse_args(argv)

    msgs = load_messages(args.data)
    X, idx, idf = features.build_index(msgs)
    state = load_state()
    blocks, stats = select(args.query, msgs, X, idx, idf, state,
                           budget=args.budget, pool=args.pool, lam=args.lam,
                           dedup=not args.no_dedup, dedup_thr=args.dedup_thr,
                           recent=args.recent, gamma=args.gamma,
                           fresh_penalty=args.fresh, fresh_days=args.fresh_days,
                           update=not args.no_update)
    if not blocks and stats.get("reason"):
        print(stats["reason"])
        return 1
    print_result(blocks, stats, msgs, args.budget)
    return 0
