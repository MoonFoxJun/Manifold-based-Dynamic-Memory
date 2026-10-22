"""Stateful recall session: the query / refine / dig-deeper loop.

A caller (human or model) drives a small state machine:

    query  -> look at the returned memory passage
    refine -> rewrite the keywords, same time anchor
    dig    -> slide the time window backwards and look again
    finish -> stop, the passage is enough

Items already returned in the session are hard-excluded, so ``dig`` always
yields *new* content; when it returns almost nothing, the window is exhausted
and the caller should ``refine`` instead.
"""

import json
import os
import sys

from . import features
from . import recall

DAY_MS = 86400e3


class Session:
    def __init__(self, msgs, X, idx, idf, state=None, budget=2000,
                 sigma_days=12.0, window_days=0.0):
        self.msgs = msgs
        self.X, self.idx, self.idf = X, idx, idf
        self.state = state if state is not None else {}
        self.budget = budget
        self.sigma_days = sigma_days
        self.t_max = max(m["time"] for m in msgs)
        self.anchor_days = window_days
        self.query_text = None
        self.last = None
        self.shown = set()

    def _run(self):
        if self.query_text is None:
            print("(no query yet)")
            return
        center = self.t_max - self.anchor_days * DAY_MS
        blocks, stats = recall.select(
            self.query_text, self.msgs, self.X, self.idx, self.idf, self.state,
            budget=self.budget, pool=80, lam=0.55, dedup=True, dedup_thr=-1,
            recent=0.0, gamma=0.6, fresh_penalty=0.6,
            time_center=center, time_sigma_days=self.sigma_days,
            time_hard=True, update=False, exclude=self.shown, exclude_hard=True)
        self.last = (blocks, stats)
        if not blocks and stats.get("reason"):
            print(f"\n(no memory to return: {stats['reason']})")
            return
        for r, fo in blocks:
            self.shown.add(r)
            self.shown.update(fo)
        print(f"\n=== '{self.query_text[:40]}' window centre: {self.anchor_days:.0f} days ago "
              f"(+-{self.sigma_days:.0f}d) | verdict {stats.get('verdict')} "
              f"(top {stats.get('top_score', 0):.3f}, strong {stats.get('strong', 0)}) | "
              f"matched {stats.get('matched', 0)}, crowding {stats.get('crowding', 0):.1f}, "
              f"suppression {stats.get('lam_eff', 0):.2f} | {stats.get('reps', 0)} reps, "
              f"{stats.get('folded', 0)} folded, ~{stats.get('chars', 0)} chars ===")
        for r, fo in blocks:
            m = self.msgs[r]
            print(f"\n[#{m['seq']} {recall.fmt_time(m['time'])} {m['role']}]")
            print(m["text"].strip()[:400])
            if fo:
                seqs = " ".join(f"#{self.msgs[j]['seq']}" for j in sorted(fo)[:8])
                print(f"   +- {len(fo)} similar folded: {seqs}")

    def query(self, text):
        self.query_text = text
        self._run()

    def refine(self, text):
        self.query_text = text
        self._run()

    def dig(self, days=7.0):
        self.anchor_days += days
        self._run()

    def show(self):
        if self.last is None:
            print("(no result yet)")
        else:
            self._run()


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="Interactive recall session.")
    ap.add_argument("--data", default=recall.DATA)
    ap.add_argument("--budget", type=int, default=2000)
    args = ap.parse_args(argv)

    msgs = recall.load_messages(args.data)
    X, idx, idf = features.build_index(msgs)
    s = Session(msgs, X, idx, idf, recall.load_state(), budget=args.budget)

    print("commands: q <keywords> / refine <keywords> / dig [days] / show / reset / finish")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        cmd, _, rest = line.partition(" ")
        cmd = cmd.lower()
        if cmd in ("q", "query"):
            s.query(rest)
        elif cmd == "refine":
            s.refine(rest)
        elif cmd == "dig":
            try:
                d = float(rest) if rest else 7.0
            except ValueError:
                d = 7.0
            s.dig(d)
        elif cmd == "show":
            s.show()
        elif cmd == "reset":
            s.shown.clear()
            print("(cleared the already-returned set)")
        elif cmd in ("finish", "quit", "exit"):
            print("(done)")
            break
        else:
            print(f"unknown command: {cmd}")
    return 0
