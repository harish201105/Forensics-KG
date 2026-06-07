#!/usr/bin/env python3
"""Per-case overall (micro) F1 for the 28 real cases under the corrected evaluator,
to refresh the appendix per-case table. One run. Writes percase_f1.json."""
import asyncio, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings
from app.services.extraction.openai_client import OpenAIClient
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.text_extractor import TextExtractor
from app.services.evaluation.real_case_evaluator import RealCaseEvaluator

settings = get_settings()


async def main():
    oc = OpenAIClient(settings)
    ev = RealCaseEvaluator(settings.data_dir, TextExtractor(oc, ForensicsOntologySchema(settings.ontology_path)))
    cases = ev.list_cases()
    rows = []
    for c in cases:
        r = await ev.evaluate_one(c)
        tp = sum(m.true_positives for m in r.entity_metrics.values())
        fp = sum(m.false_positives for m in r.entity_metrics.values())
        fn = sum(m.false_negatives for m in r.entity_metrics.values())
        p = tp / (tp + fp) if (tp + fp) else 0.0
        rr = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * p * rr / (p + rr) if (p + rr) else 0.0
        rows.append({"case_id": c["case_id"], "title": c.get("title", c["case_id"]), "f1": round(f1, 2)})
        print(f"  {c['case_id']:<22} F1={f1:.2f}", flush=True)
    rows.sort(key=lambda x: -x["f1"])
    mean_f1 = round(sum(r["f1"] for r in rows) / len(rows), 2)
    out = {"n_cases": len(rows), "mean_f1": mean_f1, "rows": rows}
    (Path(__file__).parent / "percase_f1.json").write_text(json.dumps(out, indent=2))
    print(f"\nmean per-case F1 = {mean_f1}  (range {rows[-1]['f1']}--{rows[0]['f1']})")


if __name__ == "__main__":
    asyncio.run(main())
