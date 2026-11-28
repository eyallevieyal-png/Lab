from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.simulator import compare_controllers, generate_research_report, smoke_run


def load_yaml_config(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Adaptive AI infrastructure simulator CLI")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="YAML config path")
    parser.add_argument("--smoke", action="store_true", help="Run the smoke benchmark only")
    parser.add_argument("--report", action="store_true", help="Generate the research report only")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.smoke:
        smoke_run("experiments/results/smoke_summary.json")
        print("Smoke benchmark completed")
        return

    if args.report:
        generate_research_report("experiments/results/research_report.json")
        print("Research report generated")
        return

    smoke_run("experiments/results/smoke_summary.json")
    generate_research_report("experiments/results/research_report.json")
    print("Completed smoke benchmark and research report generation")


if __name__ == "__main__":
    main()
