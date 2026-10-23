"""Server-only analysis of exported receipts; starts no model or GPU job."""
import argparse
import json
from pathlib import Path
import sys

if Path(sys.executable).resolve() != Path("/home/workspace/lrh/miniconda3/envs/equivcompiler/bin/python").resolve():
    raise RuntimeError("EvoSPE recomputation requires dbcloud/equivcompiler")
sys.path.insert(0, str(Path(__file__).parent / "analysis_source"))
from experiments.real_intervention_v6.report import analyze

parser = argparse.ArgumentParser()
parser.add_argument("--root", type=Path, required=True)
args = parser.parse_args()
if "studies" in json.loads((args.root / "V6_METRICS.json").read_text()):
    raise ValueError("select one study root: independent studies are never pooled")
rows = [json.loads(line) for line in (args.root / "PAIRS.jsonl").read_text().splitlines()]
print(json.dumps(analyze(args.root, rows), indent=2))
