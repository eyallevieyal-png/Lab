import copy
import json
import math
import os

import pytest

from experiments.aggregate_gpu_repeats import aggregate, check_comparable, main, stat

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name):
    with open(os.path.join(ROOT, "experiments", "results", name), encoding="utf-8") as fh:
        return json.load(fh)


def scaled(run, f):
    r = copy.deepcopy(run)
    r["sections"]["bandwidth"]["result"]["bandwidth_Bps"] *= f
    r["sections"]["peak"]["result"]["peak_flops"]["fp16"] *= f
    return r


def test_stat_values():
    s = stat([10.0, 12.0, 14.0])
    assert s["mean"] == 12.0 and s["sd"] == pytest.approx(2.0) and s["ci95"] == pytest.approx(4.303 * 2.0 / math.sqrt(3))
    assert math.isnan(stat([1.0])["ci95"])


def test_aggregate_synthetic_repeats_of_the_real_t4_run():
    base = load("measured_gpu_tesla_t4.json")
    runs = [scaled(base, f) for f in (0.98, 1.0, 1.02)]
    agg = aggregate(runs)
    assert agg["n_runs"] == 3 and agg["gpu"] == "Tesla T4"
    assert agg["bandwidth_GBps"]["mean"] == pytest.approx(base["sections"]["bandwidth"]["result"]["bandwidth_Bps"] / 1e9, rel=1e-6)
    assert agg["bandwidth_GBps"]["cv"] > 0
    assert "1" in agg["gemm_measured_over_predicted_by_batch"]
    assert agg["kv_transfer_GBps_by_ctx"] and all("h2d" in v and "d2h" in v for v in agg["kv_transfer_GBps_by_ctx"].values())


def test_refuses_to_mix_gpus_quick_runs_and_single_runs():
    full, quick = load("measured_gpu_tesla_t4.json"), load("measured_gpu_tesla_t4_quick.json")
    assert any("quick" in p for p in check_comparable([full, quick]))
    assert check_comparable([full, quick], allow_quick=True) == []
    other = copy.deepcopy(full)
    other["meta"]["gpu"] = "NVIDIA H100"
    assert any("gpu" in p for p in check_comparable([full, other]))
    assert any("at least 2" in p for p in check_comparable([full]))


def test_cli_refuses_then_aggregates(tmp_path, capsys):
    f1, f2 = tmp_path / "a.json", tmp_path / "b.json"
    base = load("measured_gpu_tesla_t4.json")
    f1.write_text(json.dumps(base)), f2.write_text(json.dumps(scaled(base, 1.01)))
    assert main([str(f1)]) == 2
    out = tmp_path / "agg.json"
    assert main([str(f1), str(f2), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["n_runs"] == 2
