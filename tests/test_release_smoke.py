from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_release_declares_final_method_and_named_public_package_status():
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    assert manifest["final_method"].endswith("v1.12.1")
    assert manifest["status"] == "PUBLIC_NAMED_GITHUB_UPLOAD_PACKAGE"


def test_anonymization_scan_is_clean():
    scan = json.loads((ROOT / "anonymization_scan.json").read_text())
    assert scan["status"] == "PASS"
    assert scan["issue_count"] == 0


def test_private_human_evidence_is_absent():
    files = {path.name for path in ROOT.rglob("*") if path.is_file()}
    forbidden = {
        "advisor_authorization_evidence_20260812.jpg",
        "advisor_execution_authorization_completed_v1.json",
        "human_roles_declaration_completed_v1.json",
        "annotator_a_completed_packet_v1.json",
        "annotator_b_completed_packet_v1.json",
    }
    assert not (files & forbidden)
