# SPDX-License-Identifier: Apache-2.0
"""Freeze a public Git code corpus and local tokenizer for throughput tests.

This is a length/protocol replica, explicitly not the paper's original corpus.
Only committed blobs are read; working-tree modifications are excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("tokenizer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--decode-input-tokens", type=int, choices=(512, 2048), default=512)
    args = parser.parse_args()
    if args.samples < 64:
        parser.error("Need at least 64 independent prompt offsets")

    def git(*command):
        return subprocess.check_output(["git", "-C", str(args.repository), *command])

    revision = git("rev-parse", "--verify", args.revision + "^{commit}").decode().strip()
    names = git("ls-tree", "-r", "--name-only", revision).decode().splitlines()
    files, pieces = [], []
    for name in names:
        if Path(name).suffix not in (".py", ".cpp", ".c", ".h"):
            continue
        content = git("show", f"{revision}:{name}")
        if not 512 <= len(content) <= 128 * 1024:
            continue
        try:
            text = content.decode()
        except UnicodeDecodeError:
            continue
        files.append(dict(path=name, sha256=hashlib.sha256(content).hexdigest()))
        pieces.append(f"\n# Source: {name}\n{text}\n")
        if sum(len(p) for p in pieces) >= args.samples * 8192 * 6:
            break
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    ids = tokenizer.encode("".join(pieces), add_special_tokens=False)
    if len(ids) < args.samples * 8192:
        raise ValueError("Corpus is too small for distinct non-overlapping 8K windows")
    tokenizer_files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in args.tokenizer.iterdir()
        if p.is_file() and ("token" in p.name or "encoding" in p.name or p.name == "config.json")
    }
    corpus = dict(source=args.source_url,
                  revision=revision,
                  source_files=files,
                  tokenizer_files=tokenizer_files,
                  tokenizer_sha256=hashlib.sha256(json.dumps(tokenizer_files, sort_keys=True).encode()).hexdigest(),
                  same_paper_corpus=False,
                  recipe="committed code, sorted file order, disjoint 8192-token windows",
                  prefill=[],
                  decode=[])
    bos = [] if tokenizer.bos_token_id is None else [tokenizer.bos_token_id]
    for row in range(args.samples):
        for kind, length in (("prefill", 8192), ("decode", args.decode_input_tokens)):
            start = row * 8192
            tokens = bos + ids[start:start + length - len(bos)]
            corpus[kind].append(dict(id=f"{revision[:12]}-{row}-{length}", token_ids=tokens))
    # Never silently overwrite a previously frozen corpus.
    with args.output.open("x") as output:
        json.dump(corpus, output, indent=2)


if __name__ == "__main__":
    main()
