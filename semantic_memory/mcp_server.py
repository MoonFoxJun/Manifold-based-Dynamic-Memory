"""Minimal MCP (Model Context Protocol) server over stdio.

Implements JSON-RPC 2.0 with the methods MCP clients need:

    initialize            -> server info + capabilities
    notifications/initialized
    tools/list            -> recall / remember / stats
    tools/call            -> dispatch
    ping

Tools
-----
``recall(query, budget, recent, days_ago)``
    Read: return an extractive, ordered, de-duplicated memory passage.
``remember(text, role)``
    Write: append a message to the archive and rebuild the index.
``stats()``
    Archive size, date range, and the number of retrievable messages.

Transport: newline-delimited JSON on stdin/stdout (the MCP stdio transport).

Run::

    python -m semantic_memory.mcp_server --data data/messages.jsonl
"""

import argparse
import json
import os
import sys
import time

from . import features
from . import recall

DAY_MS = 86400e3
PROTOCOL = "2024-11-05"

TOOLS = [
    {
        "name": "recall",
        "description": "Recall memory: given keywords, return a short ordered passage "
                       "of relevant messages with provenance ids. Refuses honestly when "
                       "nothing relevant exists.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string",
                          "description": "keywords (3-8 terms), not a full sentence"},
                "budget": {"type": "integer", "default": 2000,
                           "description": "maximum characters to return"},
                "recent": {"type": "number", "default": 0.5,
                           "description": "recency exponent, 0 ignores time"},
                "days_ago": {"type": "number",
                             "description": "centre the time window this many days before "
                                            "the newest message (0 = latest)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember",
        "description": "Write a message into the memory archive so future recalls can find it.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "role": {"type": "string", "enum": ["user", "assistant", "note"],
                         "default": "note"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "stats",
        "description": "Report archive size and date range.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


class Service:
    def __init__(self, data_path):
        self.path = data_path
        self.msgs = []
        self.X = self.idx = self.idf = None
        self.state = recall.load_state()
        self.reload()

    def reload(self):
        self.msgs = recall.load_messages(self.path) if os.path.exists(self.path) else []
        if self.msgs:
            self.X, self.idx, self.idf = features.build_index(self.msgs)

    def recall(self, query, budget=2000, recent=0.5, days_ago=None):
        if not self.msgs:
            return "The memory archive is empty."
        blocks, stats = recall.select(
            query, self.msgs, self.X, self.idx, self.idf, self.state,
            budget=int(budget), pool=80, dedup=True, dedup_thr=-1,
            recent=float(recent), gamma=0.6, fresh_penalty=0.6,
            time_center=(self.msgs[-1]["time"] - float(days_ago) * DAY_MS)
            if days_ago is not None else None,
            time_hard=days_ago is not None, update=True)
        if not blocks and stats.get("reason"):
            return f"No relevant memory. ({stats['reason']})"
        lines = [f"verdict={stats.get('verdict')} top={stats.get('top_score', 0):.3f} "
                 f"items={stats.get('reps', 0)} folded={stats.get('folded', 0)} "
                 f"chars={stats.get('chars', 0)}"]
        for r, fo in blocks:
            m = self.msgs[r]
            lines.append(f"\n[#{m['seq']} {recall.fmt_time(m['time'])} {m['role']}]")
            lines.append(m["text"].strip())
            if fo:
                seqs = ", ".join(f"#{self.msgs[j]['seq']}" for j in sorted(fo)[:12])
                lines.append(f"(+ {len(fo)} similar folded: {seqs})")
        return "\n".join(lines)

    def remember(self, text, role="note"):
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        seq = (max((m["seq"] for m in self.msgs), default=0) + 1) if self.msgs else 1
        entry = {"seq": seq, "role": role, "text": text,
                 "time": int(time.time() * 1000)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.msgs.append(entry)
        self.X, self.idx, self.idf = features.build_index(self.msgs)
        return f"stored message #{seq} ({len(text)} chars); archive now has {len(self.msgs)} messages"

    def stats(self):
        if not self.msgs:
            return "The memory archive is empty."
        t0, t1 = min(m["time"] for m in self.msgs), max(m["time"] for m in self.msgs)
        return (f"messages={len(self.msgs)} chars={sum(len(m['text']) for m in self.msgs)} "
                f"range={recall.fmt_time(t0)} .. {recall.fmt_time(t1)}")


def respond(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    print(json.dumps(out, ensure_ascii=False), flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description="MCP stdio server for semantic memory.")
    ap.add_argument("--data", default=recall.DATA)
    args = ap.parse_args(argv)
    svc = Service(args.data)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = req.get("method")
        msg_id = req.get("id")

        if method == "initialize":
            respond(msg_id, {
                "protocolVersion": PROTOCOL,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "semantic-memory", "version": "0.1.0"},
            })
        elif method == "notifications/initialized":
            continue
        elif method == "ping":
            respond(msg_id, {})
        elif method == "tools/list":
            respond(msg_id, {"tools": TOOLS})
        elif method == "tools/call":
            params = req.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {}) or {}
            try:
                if name == "recall":
                    text = svc.recall(arguments.get("query", ""),
                                      arguments.get("budget", 2000),
                                      arguments.get("recent", 0.5),
                                      arguments.get("days_ago"))
                elif name == "remember":
                    text = svc.remember(arguments.get("text", ""),
                                        arguments.get("role", "note"))
                elif name == "stats":
                    text = svc.stats()
                else:
                    respond(msg_id, error={"code": -32601, "message": f"unknown tool: {name}"})
                    continue
                respond(msg_id, {"content": [{"type": "text", "text": text}], "isError": False})
            except Exception as exc:  # noqa: BLE001 - report tool errors to the client
                respond(msg_id, {"content": [{"type": "text", "text": f"error: {exc}"}],
                                 "isError": True})
        else:
            respond(msg_id, error={"code": -32601, "message": f"unknown method: {method}"})


if __name__ == "__main__":
    main()
