#!/usr/bin/env python3
"""Build the reproducibility manifest for this upload package."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {
    "release_manifest.json",
    "release_manifest.json.sha256",
    "verification_report.json",
}
EXCLUDED_DIRS = {
    ".git",
    ".venv",
    "venv",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    records = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(ROOT)
        rel = rel_path.as_posix()
        if (
            rel in EXCLUDED
            or any(part in EXCLUDED_DIRS for part in rel_path.parts)
            or rel.endswith((".pyc", ".pyo"))
        ):
            continue
        records.append(
            {
                "path": rel,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )

    manifest = {
        "schema_version": "visconcord-public-release-manifest-v1.0.0",
        "status": "PUBLIC_NAMED_GITHUB_UPLOAD_PACKAGE",
        "final_method": "skillvis-v2-joint-semantic-precedence-v1.12.1",
        "release_identity": "visconcord-v1.0.0-github-upload-package-v1",
        "file_count": len(records),
        "total_bytes": sum(item["size_bytes"] for item in records),
        "files": records,
    }
    manifest_path = ROOT / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    digest = sha256_file(manifest_path)
    (ROOT / "release_manifest.json.sha256").write_text(
        f"{digest}  release_manifest.json\n", encoding="utf-8"
    )
    print(json.dumps({"file_count": len(records), "total_bytes": manifest["total_bytes"], "manifest_sha256": digest}, indent=2))


if __name__ == "__main__":
    main()
