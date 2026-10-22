"""Convert a decompressed chat-session export into the archive format.

Input:  a JSONL file with one session event per line (fields ``type``,
        ``seq``, ``time`` and ``data``), e.g. a decompressed DSH session log.
Output: ``data/messages.jsonl`` with one message per line::

        {"seq": 8, "role": "user", "text": "...", "time": 1787579841766}

The raw session files are zstd-compressed, multi-frame JSONL.  Decompress them
first with any zstd tool; with Node.js (v22+) this one-liner works::

    node -e "const z=require('zlib'),fs=require('fs');const chunks=[];\
    fs.createReadStream(process.argv[1]).pipe(z.createZstdDecompress())\
    .on('data',c=>chunks.push(c)).on('end',()=>fs.writeFileSync(process.argv[2],\
    Buffer.concat(chunks)))" session.v4.jsonl.zstd session.jsonl

Then::

    python scripts/import_chat.py session.jsonl data/messages.jsonl
"""

import argparse
import json
import os


def text_of(event):
    if event["type"] == "user/message":
        blocks = event["data"].get("content", [])
    else:
        blocks = event["data"].get("message", {}).get("content", [])
    return "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def convert(path):
    events = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    events.sort(key=lambda e: e.get("seq", 0))
    out = []
    for e in events:
        if e.get("type") in ("user/message", "assistant/message"):
            text = text_of(e)
            if text.strip():
                out.append({
                    "seq": e.get("seq"),
                    "role": "user" if e["type"] == "user/message" else "assistant",
                    "text": text,
                    "time": e.get("time", 0),
                })
    return out


def main():
    ap = argparse.ArgumentParser(description="Import a session export as a message archive.")
    ap.add_argument("input", help="decompressed session JSONL")
    ap.add_argument("output", nargs="?", default="data/messages.jsonl")
    args = ap.parse_args()

    msgs = convert(args.input)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for m in msgs:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"wrote {len(msgs)} messages to {args.output}")


if __name__ == "__main__":
    main()
