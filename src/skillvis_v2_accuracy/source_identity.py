"""Content identities for the isolated SkillVIS v2 implementation.

The project checkout is not assumed to be a Git worktree. Content hashes are
therefore first-class identities and Git fields explicitly report
unavailability instead of inventing a commit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_sha256(path: Path) -> str:
    """Match the frozen E-S1 source-tree hashing algorithm."""

    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def selected_files_sha256(root: Path, files: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for item in sorted(files):
        relative = item.relative_to(root)
        digest.update(str(relative).encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def git_identity(root: Path) -> dict[str, object]:
    try:
        inside = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "repository_state": "not_git_repository",
            "git_commit": None,
            "dirty_state": None,
            "identity_limitation": (
                "No .git metadata is available in this workspace; commit and dirty "
                "state cannot be recovered. Content hashes are recorded instead."
            ),
        }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "repository_state": "git_worktree" if inside == "true" else "unknown",
        "git_commit": commit,
        "dirty_state": bool(status.strip()),
        "dirty_entry_count": len(status.splitlines()),
    }


def validate_artifact_records(
    root: Path,
    records: Iterable[dict[str, object]],
) -> dict[str, object]:
    errors: list[dict[str, object]] = []
    checked = 0
    for record in records:
        checked += 1
        relative = str(record["path"])
        path = root / relative
        if not path.exists():
            errors.append({"path": relative, "error": "missing"})
            continue
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        if actual_hash != record["sha256"]:
            errors.append(
                {
                    "path": relative,
                    "error": "sha256_mismatch",
                    "expected": record["sha256"],
                    "actual": actual_hash,
                }
            )
        if actual_size != record["size_bytes"]:
            errors.append(
                {
                    "path": relative,
                    "error": "size_mismatch",
                    "expected": record["size_bytes"],
                    "actual": actual_size,
                }
            )
    return {
        "checked_count": checked,
        "error_count": len(errors),
        "all_pass": not errors,
        "errors": errors,
    }


def build_source_identity(root: Path) -> dict[str, object]:
    formal_run = (
        root
        / "outputs/system_comparison/es1_nl4dv/formal_run/"
        "es1-formal-v1-20260723-r1"
    )
    run_manifest_path = formal_run / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text())
    frozen_paths = {
        "protocol_v1": root / "experiments/phase3/protocol/phase3_protocol_v1.json",
        "formal_slice_v1": (
            root
            / "experiments/system_comparison/es1_nl4dv/data/"
            "formal_evaluation_slice_v1.json"
        ),
        "gold_v1": (
            root
            / "experiments/system_comparison/es1_nl4dv/data/gold_annotated_v1.json"
        ),
        "formal_run_manifest": run_manifest_path,
    }
    expected = {
        "protocol_v1": "ed9b1c18bb68f7bc72be2d9fc8591912d9b01e16fc3eb135f21a90a141ee59ed",
        "formal_slice_v1": "0f6a67e32f028383cde5967a1b6e90c93e08e89849c81e119431eb6f6ee1a9c3",
        "gold_v1": "73b6a0d9ae04907e503634d6b97040935e87f16e75a58b64bd4a4a6ae22e7ab4",
        "formal_run_manifest": "3f081509b62d84fdb48a6a1977be38eb584e264aec9465b1f2e291ef08ebcb03",
    }
    frozen = {}
    for name, path in frozen_paths.items():
        actual = sha256_file(path)
        frozen[name] = {
            "path": str(path.relative_to(root)),
            "expected_sha256": expected[name],
            "actual_sha256": actual,
            "pass": actual == expected[name],
        }

    v2_files = [
        *sorted((root / "src/skillvis_v2").glob("*.py")),
        *sorted((root / "configs/skillvis_v2").rglob("*.json")),
    ]
    formal_validation = validate_artifact_records(root, run_manifest["artifacts"])
    diagnostic_records = json.loads(
        (root / "data/skillvis_v2/es1_v1_diagnostic_cases.json").read_text()
    )
    frozen_formal_source_hash = str(
        diagnostic_records["records"][0]["skillvis_v1_prediction"]["system_version"]
    ).removeprefix("source-tree:")
    current_v1_tree_hash = tree_sha256(root / "src/skillvis")
    return {
        "schema_version": "skillvis-v2-source-identity-v0.1.0",
        "git": git_identity(root),
        "source": {
            "v1_source_tree_path": "src/skillvis",
            "v1_source_tree_sha256": current_v1_tree_hash,
            "v1_formal_run_source_tree_sha256": frozen_formal_source_hash,
            "v1_source_tree_matches_formal_run": (
                current_v1_tree_hash == frozen_formal_source_hash
            ),
            "source_hash_scope_note": (
                "This all-file hash matches the frozen formal runner algorithm "
                "and therefore includes bytecode caches. Frozen asset hashes and "
                "the full v1 test suite are the primary immutability checks."
            ),
            "v2_selected_source_and_config_sha256": selected_files_sha256(
                root, v2_files
            ),
            "v2_selected_file_count": len(v2_files),
            "v2_selected_files": [str(path.relative_to(root)) for path in v2_files],
        },
        "frozen_assets": frozen,
        "frozen_assets_all_pass": all(item["pass"] for item in frozen.values()),
        "formal_run_artifact_revalidation": formal_validation,
    }
