"""Lossless macOS filesystem compression of generated project files; dry-run by default."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def compress(path):
    before = path.lstat()
    if path.is_symlink() or before.st_nlink != 1:
        return None
    checksum = digest(path)
    with tempfile.TemporaryDirectory(prefix=".storage-opt-", dir=path.parent) as temp:
        target = Path(temp) / path.name
        subprocess.run(
            ["/usr/bin/ditto", "--hfsCompression", "--nocache", str(path), str(target)],
            check=True,
            capture_output=True,
        )
        after = target.stat()
        if digest(target) != checksum:
            raise ValueError(f"Compression checksum mismatch: {path}")
        current = path.lstat()
        if (current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns) != (
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ):
            raise ValueError(f"File changed during compression: {path}")
        if after.st_blocks >= before.st_blocks:
            return None
        if (after.st_mode, after.st_uid, after.st_gid, after.st_mtime_ns) != (
            before.st_mode,
            before.st_uid,
            before.st_gid,
            before.st_mtime_ns,
        ):
            raise ValueError(f"Compression changed file metadata: {path}")
        with target.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(target, path)
        if digest(path) != checksum:
            raise ValueError(f"Post-replacement checksum mismatch: {path}")
        return {
            "path": str(path),
            "sha256": checksum,
            "saved_bytes": (before.st_blocks - after.st_blocks) * 512,
        }


def run(root, apply=False, limit=None):
    if platform.system() != "Darwin":
        raise ValueError("Transparent compression requires macOS")
    candidates = []
    for name in ("artifacts", "data"):
        folder = root / name
        if folder.is_symlink():
            raise ValueError("Refusing a symlinked data/artifact root")
        for path in folder.rglob("*"):
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                continue
            stat = path.stat()
            if (
                path.suffix in {".json", ".jsonl", ".html", ".csv", ".txt", ".log"}
                and stat.st_size >= 65536
                and not stat.st_flags & 0x20
            ):
                candidates.append(path)
    candidates.sort(key=lambda p: p.stat().st_size, reverse=True)
    if limit is not None:
        candidates = candidates[:limit]
    result = {"candidate_files": len(candidates), "applied": apply, "saved_bytes": 0, "files": []}
    if apply:
        for path in candidates:
            row = compress(path)
            if row:
                result["files"].append(row)
                result["saved_bytes"] += row["saved_bytes"]
        directory = root / "artifacts" / "storage-audit"
        directory.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="compression-", suffix=".json", dir=directory, delete=False
        ) as stream:
            json.dump(result, stream, indent=2)
        result["audit"] = stream.name
    print(json.dumps({k: v for k, v in result.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(Path(__file__).resolve().parents[1], args.apply, args.limit)
