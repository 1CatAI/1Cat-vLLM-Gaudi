# SPDX-License-Identifier: Apache-2.0
"""Freeze current V4.1 PR metadata, checks and diffs before porting an implementation."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess


PRS = {
    "vllm-project/vllm": (56228, 56214, 56215, 56255, 56266, 53577, 55654, 56254, 56227, 56219, 56220),
    "sgl-project/sglang": (38798, 38848, 38879, 38861, 37382),
}


def api(endpoint, *, paginate=False):
    command = ["gh", "api", endpoint]
    if paginate:
        command.append("--paginate")
    raw = subprocess.check_output(command, text=True, timeout=180)
    decoder, pages = json.JSONDecoder(), []
    while raw.strip():
        raw = raw.lstrip()
        value, end = decoder.raw_decode(raw)
        pages.append(value)
        raw = raw[end:]
    return [item for page in pages for item in page] if paginate else pages[0]


def save(path, value):
    data = (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def collect(spec, output):
    repo, number = spec
    metadata = api(f"repos/{repo}/pulls/{number}")
    head = metadata["head"]["sha"]
    stem = f"{repo.replace('/', '_')}-{number}"
    files = api(f"repos/{repo}/pulls/{number}/files?per_page=100", paginate=True)
    checks = {}
    for kind, endpoint in (
        ("status", f"repos/{repo}/commits/{head}/status"),
        ("check_runs", f"repos/{repo}/commits/{head}/check-runs?per_page=100"),
    ):
        try:
            checks[kind] = api(endpoint)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            checks[kind] = {"unavailable": str(error)}
    save(output / f"{stem}.json", metadata)
    diff_hash = save(output / f"{stem}-files.json", files)
    save(output / f"{stem}-checks.json", checks)
    # Read the head again: never qualify a diff/check bundle against a moving head.
    after = api(f"repos/{repo}/pulls/{number}")
    if after["head"]["sha"] != head:
        raise RuntimeError(f"{repo}#{number} changed during capture; rerun the audit")
    result = {
        "repository": repo, "pull_request": number, "url": metadata["html_url"],
        "state": metadata["state"], "merged": metadata["merged"], "draft": metadata["draft"],
        "head_sha": head, "base_sha": metadata["base"]["sha"],
        "merge_commit_sha": metadata.get("merge_commit_sha"),
        "changed_files": len(files), "diff_sha256": diff_hash, "ci": checks,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
    }
    print(f"{repo}#{number}: {head[:12]} {result['state']} merged={result['merged']} files={len(files)}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    searches = {}
    for repo in PRS:
        for query in ("DeepSeek V4.1", "DSpark pipeline", "Engram"):
            command = ["gh", "search", "prs", query, "--repo", repo, "--sort", "updated", "--limit", "40",
                       "--json", "number,title,url,state,isDraft,updatedAt"]
            searches[f"{repo}:{query}"] = json.loads(subprocess.check_output(command, text=True, timeout=90))
    save(args.output / "searches.json", searches)
    specs = [(repo, number) for repo, numbers in PRS.items() for number in numbers]
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda spec: collect(spec, args.output), specs))
    save(args.output / "upstream-lock.json", {"schema_version": 1, "pull_requests": results})


if __name__ == "__main__":
    main()
