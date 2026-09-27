#!/usr/bin/env python3
"""Turn a Jupyter notebook into a compact, token-cheap .py that is quick to read.

- keeps code cells and markdown (as comments)
- keeps plain-text outputs, truncated per cell (widget state, images, HTML dropped)
- REDACTS secrets (Hugging Face / AWS / GitHub tokens, private keys) in code and outputs

usage: python tools/extract_notebook.py notebooks/v1.ipynb docs/v1_eda.py [--max-out 40]
"""
import argparse
import json
import re
import sys

SECRETS = [
    (re.compile(r"hf_[A-Za-z0-9]{20,}"), "<REDACTED_HF_TOKEN>"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "<REDACTED_AWS_KEY_ID>"),
    (re.compile(r"sk-[A-Za-z0-9_-]{20,}"), "<REDACTED_API_KEY>"),
    (re.compile(r"(?i)(aws_secret_access_key\s*[=:]\s*)\S+"), r"\1<REDACTED>"),
    (re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"), "<REDACTED_GH_TOKEN>"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
     "<REDACTED_PRIVATE_KEY>"),
]


def redact(text):
    n = 0
    for pat, rep in SECRETS:
        text, k = pat.subn(rep, text)
        n += k
    return text, n


def text_of(output):
    if output.get("output_type") == "stream":
        return "".join(output.get("text", ""))
    data = output.get("data", {})
    if "text/plain" in data:
        return "".join(data["text/plain"])
    if output.get("output_type") == "error":
        return f"{output.get('ename')}: {output.get('evalue')}"
    return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--max-out", type=int, default=40, help="max output lines kept per cell")
    a = ap.parse_args()

    nb = json.load(open(a.src, encoding="utf-8"))
    parts, redactions = [f"# Extracted from {a.src} (code + truncated text outputs; secrets redacted)\n"], 0
    for i, cell in enumerate(nb.get("cells", [])):
        src = "".join(cell.get("source", ""))
        if cell["cell_type"] == "markdown":
            parts.append("\n".join("# " + l for l in src.splitlines()) + "\n")
            continue
        if cell["cell_type"] != "code":
            continue
        parts.append(f"\n# %% [cell {i}]\n{src}\n")
        out = "\n".join(t for t in (text_of(o) for o in cell.get("outputs", [])) if t)
        # strip ANSI escapes and pip progress bars
        out = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", out)
        lines = [l for l in out.splitlines() if l.strip() and "━━" not in l]
        if lines:
            kept = lines[: a.max_out]
            more = f"\n# ... ({len(lines) - len(kept)} more output lines truncated)" if len(lines) > len(kept) else ""
            parts.append("# --- output ---\n" + "\n".join("# " + l for l in kept) + more + "\n")
    text, redactions = redact("".join(parts))
    open(a.dst, "w", encoding="utf-8").write(text)
    print(f"wrote {a.dst}: {len(text):,} chars (~{len(text) // 4:,} tokens) from {len(open(a.src, encoding='utf-8').read()):,} "
          f"chars of notebook JSON; secrets redacted: {redactions}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
