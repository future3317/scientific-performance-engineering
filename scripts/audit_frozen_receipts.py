"""Write a fresh historical diagnosis directory; never write into source evidence."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark.formal.failure_funnel import audit_root, verify_frozen_counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--frozen-audit", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("diagnostic output must be separate from historical evidence")
    rows, summary = audit_root(source)
    verify_frozen_counts(summary, json.loads(args.frozen_audit.read_text()))
    output.mkdir(parents=True, exist_ok=False)
    with (output / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].to_dict()))
        writer.writeheader()
        writer.writerows(row.to_dict() for row in rows)
    groups = Counter((row.arm, row.family, row.stage) for row in rows)
    with (output / "grouped.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["arm", "family", "stage", "count"])
        writer.writerows((*key, count) for key, count in sorted(groups.items()))
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Historical failure funnel", "", "Retrospective, provider-free; original files read only.", "",
             f"Scheduled {summary['observed_cells']}; execution-valid {summary['execution_valid']}; verifier records {summary['verifier_records']}.", "",
             "| Stage | Cells |", "|---|---:|"]
    lines += [f"| {stage} | {count} |" for stage, count in summary["stages"].items()]
    lines += ["", "20 nonlaunches remain missing outcomes, never zero outcomes.", "",
              "## Attribution and next action", "",
              "Observed: controller terminal reasons, API errors, anti-cheat failures and unverified speedups are distinct failure paths.",
              "Hypotheses: context growth contributes to token exhaustion; planning/finalization contributes to turn exhaustion. These causes require new controlled tests.",
              "Historical empty tool traces do not establish absence of tool use. Prompt-section attribution and final five turns are unavailable where not persisted; marked missing, not reconstructed.", "",
              "## Representative cells", ""]
    for reason in sorted({r.failure_code for r in rows if r.failure_code}):
        row = next(r for r in rows if r.failure_code == reason)
        lines.append(f"- {reason}: {row.arm}/{row.outer}/{row.task_id}; source `{row.source}`; tools={row.tool_calls}, trace rows={row.tool_trace_count}.")
    (output / "FAILURE_FUNNEL.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "source_digests"}, indent=2))


if __name__ == "__main__":
    main()
