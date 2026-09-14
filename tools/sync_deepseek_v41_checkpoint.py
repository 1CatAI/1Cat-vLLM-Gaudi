# SPDX-License-Identifier: Apache-2.0
"""Resume a pinned checkpoint download and validate files without expanding tensors."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
import urllib.request


MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
REVISION = "dba1be0a40aa45a94ad051997016db3960a90277"


def parallel_download(url: str, path: Path, size: int, workers: int):
    """Resume validated HTTP ranges in one sparse file, without extra shard copies."""
    chunk_bytes = 64 * 2**20
    progress_path = path.with_name(path.name + ".ranges.json")
    if progress_path.exists():
        state = json.loads(progress_path.read_text())
        if state["url"] != url or state["bytes"] != size or state["chunk_bytes"] != chunk_bytes:
            raise ValueError("Partial download belongs to a different pinned file")
    else:
        # Adopt the contiguous prefix produced by an earlier curl download.
        prefix = path.stat().st_size if path.exists() else 0
        if prefix > size:
            raise ValueError("Partial download is longer than the pinned source")
        state = {"url": url, "bytes": size, "chunk_bytes": chunk_bytes,
                 "complete": list(range(prefix // chunk_bytes))}
        # Publish before truncate: after interruption a sparse logical length
        # must never be confused with downloaded bytes.
        temporary = progress_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state))
        temporary.replace(progress_path)
    complete = set(state["complete"])
    lock = threading.Lock()
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(fd, size)

    def download(index):
        if index in complete:
            return
        begin, end = index * chunk_bytes, min(size, (index + 1) * chunk_bytes) - 1
        for attempt in range(6):
            try:
                request = urllib.request.Request(url + f"?gaudi_range={begin}-{end}&attempt={attempt}",
                                                  headers={"Range": f"bytes={begin}-{end}"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    expected = f"bytes {begin}-{end}/{size}"
                    if response.status != 206 or response.headers.get("Content-Range") != expected:
                        raise RuntimeError("Server did not honor the exact immutable byte range")
                    position = begin
                    while position <= end:
                        data = response.read(min(4 * 2**20, end - position + 1))
                        if not data:
                            raise RuntimeError("Truncated HTTP range")
                        remaining = memoryview(data)
                        while remaining:
                            count = os.pwrite(fd, remaining, position)
                            if not count:
                                raise OSError("Could not persist checkpoint range")
                            position += count
                            remaining = remaining[count:]
                    if response.read(1):
                        raise RuntimeError("HTTP range exceeded the requested end")
                with lock:
                    os.fdatasync(fd)
                    complete.add(index)
                    state["complete"] = sorted(complete)
                    temporary = progress_path.with_suffix(".tmp")
                    temporary.write_text(json.dumps(state))
                    temporary.replace(progress_path)
                    if len(complete) % 32 == 0:
                        print(f"ranges {path.name}: {len(complete) * chunk_bytes / 2**30:.1f}/{size / 2**30:.1f} GiB",
                              flush=True)
                return
            except (OSError, RuntimeError) as error:
                if attempt == 5:
                    raise
                print(f"retry {path.name} range={index} attempt={attempt + 1}: {error}", flush=True)
                time.sleep(min(2**attempt, 16))
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(download, range((size + chunk_bytes - 1) // chunk_bytes)))
    finally:
        os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--range-workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1 or args.range_workers < 1:
        parser.error("Download worker counts must be positive")
    args.model.mkdir(parents=True, exist_ok=True)
    args.evidence.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(
        f"https://huggingface.co/api/models/{MODEL}/revision/{args.revision}?blobs=true", timeout=60
    ) as response:
        info = json.load(response)
    if info["sha"] != args.revision:
        raise RuntimeError("The model API did not resolve the requested immutable revision")
    (args.evidence / "remote-model.json").write_text(json.dumps(info, indent=2) + "\n")
    files = info["siblings"]
    def resident(entry):
        path = args.model / entry["rfilename"]
        if path.is_file() and path.stat().st_size == entry["size"]:
            return entry["size"]
        partial = path.with_name(path.name + ".dsv41-download")
        return min(entry["size"], partial.stat().st_blocks * 512) if partial.exists() else 0
    missing_bytes = sum(entry["size"] - resident(entry) for entry in files)
    if shutil.disk_usage(args.model).free < missing_bytes + 8 * 2**30:
        raise RuntimeError(f"Checkpoint destination requires {missing_bytes / 2**30:.2f} GiB plus 8 GiB reserve")
    pid = {"pid": os.getpid(), "pgid": os.getpgrp(), "started_at": datetime.now(timezone.utc).isoformat()}
    (args.evidence / "owned-process.json").write_text(json.dumps(pid, indent=2) + "\n")

    def sync(entry):
        relative = Path(entry["rfilename"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe checkpoint filename")
        path = args.model / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        record = args.evidence / "files" / (str(relative).replace("/", "_") + ".json")
        if record.exists() and path.exists():
            cached, stat = json.loads(record.read_text()), path.stat()
            same_source = (cached.get("blob_id") == entry["blobId"] or
                           (bool(entry.get("lfs")) and cached.get("sha256") == entry["lfs"]["sha256"]))
            if (same_source and cached["bytes"] == stat.st_size == entry["size"]
                    and cached.get("inode") == stat.st_ino and cached.get("mtime_ns") == stat.st_mtime_ns):
                print(f"verified-unchanged {relative}", flush=True)
                return cached
        force_metadata_download = False
        if path.exists() and path.stat().st_size == entry["size"] and not entry.get("lfs"):
            blob = hashlib.sha1(b"blob " + str(entry["size"]).encode() + b"\0" + path.read_bytes()).hexdigest()
            force_metadata_download = blob != entry["blobId"]
        if force_metadata_download or not path.exists() or path.stat().st_size != entry["size"]:
            # Do not destroy a different local text revision before the pinned copy is complete.
            partial = path.with_name(path.name + ".dsv41-download")
            url = f"https://huggingface.co/{MODEL}/resolve/{args.revision}/{relative.as_posix()}"
            print(f"download {relative} {entry['size']} bytes", flush=True)
            if entry["size"] > 2**30 and args.range_workers > 1:
                parallel_download(url, partial, entry["size"], args.range_workers)
            else:
                subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "8",
                                "--retry-all-errors", "--retry-delay", "3", "--connect-timeout", "30", "--continue-at", "-",
                                "--output", str(partial), url], check=True)
            if partial.stat().st_size != entry["size"]:
                raise RuntimeError(f"Incomplete download: {relative}")
            candidate = partial
        else:
            candidate = path
        digest = hashlib.sha256()
        blob_digest = hashlib.sha1(b"blob " + str(entry["size"]).encode() + b"\0")
        with candidate.open("rb") as stream:
            while chunk := stream.read(16 * 2**20):
                digest.update(chunk)
                if not entry.get("lfs"):
                    blob_digest.update(chunk)
        expected = entry.get("lfs", {}).get("sha256")
        if (expected and digest.hexdigest() != expected) or (not expected and blob_digest.hexdigest() != entry["blobId"]):
            raise RuntimeError(f"Hash mismatch, file was not published: {relative}")
        if candidate != path:
            if path.exists():
                backup = args.evidence / "previous-local-files" / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                if not backup.exists():
                    shutil.copy2(path, backup)
            candidate.replace(path)
        stat = path.stat()
        result = {"file": str(relative), "bytes": entry["size"], "sha256": digest.hexdigest(),
                  "mtime_ns": stat.st_mtime_ns, "inode": stat.st_ino, "blob_id": entry["blobId"],
                  "revision": args.revision}
        print(f"verified {relative}", flush=True)
        # One record per file permits recovery after interruption without discarding partial evidence.
        record.parent.mkdir(exist_ok=True)
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(result) + "\n")
        temporary.replace(record)
        return result

    # Begin the two large missing files first while independent local verification proceeds.
    files.sort(key=lambda entry: (args.model / entry["rfilename"]).exists())
    verified, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(sync, entry): entry for entry in files}
        for future in as_completed(futures):
            try:
                verified.append(future.result())
            except Exception as error:
                failure = {"file": futures[future]["rfilename"], "error": str(error)}
                failed.append(failure)
                print(f"verification failed: {failure}", flush=True)
                (args.evidence / "failed.json").write_text(json.dumps(failed, indent=2) + "\n")
    if failed:
        raise RuntimeError(f"Checkpoint is not publishable: {len(failed)} file(s) failed; see failed.json")
    verified.sort(key=lambda record: record["file"])
    temporary = args.evidence / "complete.json.tmp"
    temporary.write_text(json.dumps({"revision": args.revision, "files": verified}, indent=2))
    temporary.replace(args.evidence / "complete.json")


if __name__ == "__main__":
    main()
