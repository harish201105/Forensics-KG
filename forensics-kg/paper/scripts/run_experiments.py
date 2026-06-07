#!/usr/bin/env python3
"""Real-case experiments for the paper: per-case breakdown + an ablation that
isolates the value of ontology-guided extraction.

Two conditions, scored by the IDENTICAL matcher (RealCaseEvaluator) against the
same documented-fact gold:
  FULL      - the system: ontology-guided, schema-registry, two-stage extraction.
  BASELINE  - same LLM + JSON output, but a single generic entity-extraction
              prompt with NO forensic ontology type descriptions, no per-type
              rules, and no two-stage relation grounding.

Outputs paper/scripts/experiment_results.json and prints a summary. No number is
hand-edited; whatever this prints is what goes in the paper.
"""
import asyncio, json, sys, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]            # forensics-kg/
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings
from app.services.extraction.openai_client import OpenAIClient
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.text_extractor import TextExtractor
from app.services.evaluation.real_case_evaluator import RealCaseEvaluator

settings = get_settings()

# --- Naive baseline extractor (ablation: -ontology guidance, single stage) ----
BASELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string"},
                    "properties": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "role": {"type": "string"},
                            "type": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "additionalProperties": True,
                    },
                },
                "required": ["entity_type", "properties"],
            },
        }
    },
    "required": ["entities"],
}


class BaselineExtractor:
    """Single-prompt JSON entity extractor with no ontology scaffolding."""

    def __init__(self, openai_client):
        self._openai = openai_client

    async def extract(self, text, source_type="fir", model_override=None):
        system = (
            "Extract named entities from the text and return JSON. Each entity "
            "has entity_type and properties. Use entity_type from: Person, "
            "Location, Weapon, CrimeType, TimeEvent, Case. For Person include "
            "name and role. For Location/Weapon/CrimeType include name or type. "
            "For TimeEvent include a name or description containing the date. "
            "Only extract what is explicitly stated."
        )
        res = await self._openai.extract_structured(
            system_prompt=system,
            user_prompt=f"Text:\n{text}",
            response_schema=BASELINE_SCHEMA,
            model_override=model_override,
        )
        return {"entities": res.get("entities", []), "relationships": [], "metadata": {}}


def macro(per_type):
    """Macro P/R/F1 = unweighted mean over entity types."""
    ms = list(per_type.values())
    if not ms:
        return (0.0, 0.0, 0.0)
    return (sum(m.precision for m in ms) / len(ms),
            sum(m.recall for m in ms) / len(ms),
            sum(m.f1 for m in ms) / len(ms))


def trial_record(agg):
    """Flatten one evaluate_all() into a plain dict of metrics for averaging."""
    pt = agg.per_entity_type
    maP, maR, maF = macro(pt)
    return {
        "micro": {"P": agg.overall_precision, "R": agg.overall_recall, "F1": agg.overall_f1},
        "macro": {"P": maP, "R": maR, "F1": maF},
        "per_type": {t: {"P": m.precision, "R": m.recall, "F1": m.f1}
                     for t, m in pt.items()},
        "per_case": {r.fir_id: r.aggregate_f1 for r in agg.per_case},
    }


def _ms(vals):
    return {"mean": round(statistics.mean(vals), 3),
            "sd": round(statistics.pstdev(vals), 3) if len(vals) > 1 else 0.0}


def aggregate(name, trials):
    """Mean +/- stdev across trials, per type and overall."""
    types = sorted({t for tr in trials for t in tr["per_type"]})
    out = {"n_trials": len(trials), "per_type": {}, "micro": {}, "macro": {}}
    for k in ("P", "R", "F1"):
        out["micro"][k] = _ms([tr["micro"][k] for tr in trials])
        out["macro"][k] = _ms([tr["macro"][k] for tr in trials])
    for t in types:
        out["per_type"][t] = {k: _ms([tr["per_type"].get(t, {}).get(k, 0.0) for tr in trials])
                              for k in ("P", "R", "F1")}
    # per-case mean F1 across trials
    cases = sorted({c for tr in trials for c in tr["per_case"]})
    out["per_case"] = {c: _ms([tr["per_case"].get(c, 0.0) for tr in trials]) for c in cases}

    print(f"\n===== {name}  (mean +/- sd over {len(trials)} trials) =====")
    print(f"{'type':<12}{'P':>12}{'R':>12}{'F1':>12}")
    for t in types:
        r = out["per_type"][t]
        print(f"{t:<12}" + "".join(f"{r[k]['mean']:>6.2f}±{r[k]['sd']:<5.2f}" for k in ('P','R','F1')))
    for lvl in ("micro", "macro"):
        r = out[lvl]
        print(f"{lvl.upper():<12}" + "".join(f"{r[k]['mean']:>6.2f}±{r[k]['sd']:<5.2f}" for k in ('P','R','F1')))
    return out


async def run_condition(label, evaluator, n):
    trials = []
    for i in range(n):
        print(f"  {label} trial {i+1}/{n} ...", flush=True)
        trials.append(trial_record(await evaluator.evaluate_all()))
    return aggregate(label, trials)


async def main(n=3, outfile="experiment_results.json"):
    oc = OpenAIClient(settings)
    schema = ForensicsOntologySchema(settings.ontology_path)
    full_eval = RealCaseEvaluator(settings.data_dir, TextExtractor(oc, schema))
    base_eval = RealCaseEvaluator(settings.data_dir, BaselineExtractor(oc))

    print(f"model={settings.openai_model}  cases={len(full_eval.list_cases())}  trials={n}")
    full = await run_condition("FULL (ontology-guided, two-stage)", full_eval, n)
    base = await run_condition("BASELINE (naive JSON, no ontology)", base_eval, n)

    out = {"model": settings.openai_model, "n_trials": n, "full": full, "baseline": base}
    (Path(__file__).parent / outfile).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outfile}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    outfile = sys.argv[2] if len(sys.argv) > 2 else "experiment_results.json"
    asyncio.run(main(n, outfile))
