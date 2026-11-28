"""Guards for the research-integrity structure: the active summary is current, the archive stays labelled, the registry points at real files."""
import json
import os

from research.pipeline.make_results_summary import build
from research.registry.build_registry import REGISTRY
from research.validation import verify_simulator_fixes

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLASSES = {"reproduced", "validated_simulation", "preliminary_hardware", "theoretical", "superseded", "unverified_hypothesis"}


def test_all_eight_simulator_defects_are_fixed_in_the_current_path():
    assert verify_simulator_fixes.main() == 0


def test_root_results_summary_is_current_and_self_describing():
    committed = json.load(open(os.path.join(ROOT, "results_summary.json"), encoding="utf-8"))
    meta = committed["metadata"]
    for key in ("experiment_id", "simulator_commit", "generated_utc", "config", "evidence_class", "supersedes", "regenerate"):
        assert key in meta
    assert "simulated" in meta["evidence_class"]
    assert committed["aggregate"] == json.loads(json.dumps(build()["aggregate"]))      # regenerating from the simulator gives the same numbers
    assert any(r["controller"] == "Oracle" and r["mean_regret"] == 0.0 for r in committed["summary"])


def test_archive_is_labelled_and_not_importable():
    for dirpath, _, files in os.walk(os.path.join(ROOT, "archive", "legacy_results")):
        for f in files:
            assert f.startswith("SUPERSEDED") or f.endswith(".md"), f
    assert not any(f.endswith(".py") for _, _, fs in os.walk(os.path.join(ROOT, "archive")) for f in fs)
    assert "SUPERSEDED" in open(os.path.join(ROOT, "archive", "legacy_results", "v0_2026-09-17", "SUPERSEDED.md"), encoding="utf-8").read()


def test_registry_entries_are_valid_and_files_exist():
    ids = [e["id"] for e in REGISTRY]
    assert len(ids) == len(set(ids))
    for e in REGISTRY:
        assert e["class"] in CLASSES, e["id"]
        assert e["validation_status"], e["id"]
        for o in e["outputs"]:
            assert o["exists"], (e["id"], o["path"])
    assert any(e["class"] == "superseded" for e in REGISTRY)
    committed = json.load(open(os.path.join(ROOT, "research", "EXPERIMENT_REGISTRY.json"), encoding="utf-8"))
    assert {e["id"] for e in committed["experiments"]} == set(ids)       # the committed registry is up to date


def test_no_test_writes_into_tracked_results():
    for name in ("test_report.json", "test_smoke.json"):
        assert not os.path.exists(os.path.join(ROOT, "experiments", "results", name))
