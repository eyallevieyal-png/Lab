from __future__ import annotations

from src.simulator import generate_research_report, smoke_run


if __name__ == "__main__":
    smoke_run("experiments/results/smoke_summary.json")
    generate_research_report("experiments/results/research_report.json")
    print("Smoke simulation completed. Results saved to experiments/results/smoke_summary.json")
    print("Research report generated at experiments/results/research_report.json")
