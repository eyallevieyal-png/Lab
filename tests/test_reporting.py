from src.simulator import compare_controllers, generate_research_report


def test_compare_controllers_returns_runs():
    runs = compare_controllers(
        workloads=["agentic"],
        controllers=["reactive", "adaptive"],
        precisions=["int4"],
        num_ops=6,
        seeds=[7],
    )
    assert len(runs) == 2
    assert {run.controller for run in runs} == {"Reactive", "Adaptive"}


def test_generate_research_report_writes_summary():
    report = generate_research_report("experiments/results/test_report.json")
    assert "summary" in report
    assert "controllers" in report
    assert "workloads" in report
