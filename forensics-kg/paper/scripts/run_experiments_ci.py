#!/usr/bin/env python3
"""Extended real-case measurement for the paper revision:
  * bootstrap confidence intervals (resampling the 10 cases),
  * fairer complementary metrics: principal-location recall and date coverage,
  * concrete false-positive / false-negative examples.

Full ontology-guided system only (baseline already measured in
experiment_results.json). Writes experiment_ci.json. No number is hand-edited.
"""
import asyncio, json, sys, random, re, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings
from app.services.extraction.openai_client import OpenAIClient
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.text_extractor import TextExtractor
from app.services.evaluation.evaluator import _compute_metrics, _get_extracted_by_type
from app.services.evaluation.real_case_evaluator import (
    _canon_crimes, _date_match, _date_keys, _MONTHS,
)

settings = get_settings()
N_RUNS = 3
B = 2000
random.seed(7)
TYPES = ["Person", "Location", "Weapon", "CrimeType", "TimeEvent"]


def score_case(entities, gold):
    """Per-type TP/FP/FN for one case, plus raw extracted location/time text."""
    out = {}
    # Person / Location / Weapon — fuzzy string matching (same as evaluator)
    for etype, gkey in (("Person", "persons"), ("Location", "locations"), ("Weapon", "weapons")):
        gi = gold.get(gkey, []) or []
        gi = [p[0] if isinstance(p, (list, tuple)) else str(p) for p in gi] if gkey == "persons" else [str(x) for x in gi]
        ex = _get_extracted_by_type(entities, etype)
        m = _compute_metrics(etype, gi, ex)
        out[etype] = (m.true_positives, m.false_positives, m.false_negatives)
    # CrimeType — canonical keyword sets
    gc = _canon_crimes(" ".join(gold.get("crime_types", []) or []))
    crime_text = " ".join(
        f"{e['properties'].get('title','')} {e['properties'].get('name','')} "
        f"{e['properties'].get('description','')} {e['properties'].get('crime_type','')}"
        for e in entities if e.get("entity_type") in ("Case", "CrimeType"))
    pc = _canon_crimes(crime_text)
    out["CrimeType"] = (len(gc & pc), len(pc - gc), len(gc - pc))
    # TimeEvent — date-aware matching
    gold_dates = [str(t) for t in (gold.get("time_events", []) or [])]
    ev_texts = [f"{e['properties'].get('name','')} {e['properties'].get('description','')} "
                f"{e['properties'].get('date','')}"
                for e in entities if e.get("entity_type") == "TimeEvent"]
    used, matched = set(), 0
    for g in gold_dates:
        for k, ev in enumerate(ev_texts):
            if k in used:
                continue
            if _date_match(g, ev):
                used.add(k); matched += 1; break
    out["TimeEvent"] = (matched, len(ev_texts) - matched, len(gold_dates) - matched)

    # ---- complementary, fairer metrics ----
    loc_tp = out["Location"][0]
    principal_loc_hit = 1 if loc_tp >= 1 else 0          # gold has exactly 1 location
    # date coverage: fraction of gold YEARS appearing anywhere in the timeline text
    gold_years = set()
    for g in gold_dates:
        gy, _ = _date_keys(g); gold_years |= gy
    all_ev = " ".join(ev_texts)
    ev_years, _ = _date_keys(all_ev)
    covered = len(gold_years & ev_years)
    date_cov = covered / len(gold_years) if gold_years else None
    ex_locs = [e["properties"].get("name", "") for e in entities if e.get("entity_type") == "Location"]
    return out, principal_loc_hit, date_cov, ex_locs, ev_texts, gold


def micro(counts):
    tp = sum(c[0] for c in counts.values()); fp = sum(c[1] for c in counts.values()); fn = sum(c[2] for c in counts.values())
    p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
    return 2*p*r/(p+r) if p+r else 0.0


def macro(counts):
    fs = []
    for t in counts:
        tp, fp, fn = counts[t]
        p = tp/(tp+fp) if tp+fp else 0.0; r = tp/(tp+fn) if tp+fn else 0.0
        fs.append(2*p*r/(p+r) if p+r else 0.0)
    return sum(fs)/len(fs) if fs else 0.0


async def main():
    oc = OpenAIClient(settings)
    te = TextExtractor(oc, ForensicsOntologySchema(settings.ontology_path))
    cases = [json.loads(p.read_text()) for p in sorted((Path(settings.data_dir) / "real_cases").glob("*.json"))]

    # per-case counts summed over runs; collect aux metrics from run 1
    case_counts = {c["case_id"]: {t: [0, 0, 0] for t in TYPES} for c in cases}
    ploc, dcov, examples = {}, {}, []
    for run in range(N_RUNS):
        print(f"run {run+1}/{N_RUNS}", flush=True)
        for c in cases:
            ents = (await te.extract(c["narrative"], "fir")).get("entities", [])
            counts, ph, dc, ex_locs, ev_texts, gold = score_case(ents, c["gold"])
            for t in TYPES:
                for i in range(3):
                    case_counts[c["case_id"]][t][i] += counts[t][i]
            if run == 0:
                ploc[c["case_id"]] = ph
                if dc is not None:
                    dcov[c["case_id"]] = dc
                # capture a couple of concrete examples
                if len(examples) < 6:
                    gold_locs = gold.get("locations", [])
                    fp_locs = [l for l in ex_locs if l and all(
                        gl.split(",")[0].lower() not in l.lower() and l.lower() not in gl.lower()
                        for gl in gold_locs)]
                    if fp_locs:
                        examples.append({"case": c["case_id"], "type": "Location FP",
                                         "gold": gold_locs, "spurious": fp_locs[:4]})

    # point estimates (per-case counts summed over runs -> ratios unaffected by 3x)
    agg = {t: [sum(case_counts[c][t][i] for c in case_counts) for i in range(3)] for t in TYPES}
    micro_pt = micro(agg); macro_pt = macro(agg)

    # bootstrap over the 10 cases
    ids = list(case_counts.keys())
    mic, mac = [], []
    for _ in range(B):
        samp = [random.choice(ids) for _ in ids]
        cc = {t: [0, 0, 0] for t in TYPES}
        for cid in samp:
            for t in TYPES:
                for i in range(3):
                    cc[t][i] += case_counts[cid][t][i]
        mic.append(micro(cc)); mac.append(macro(cc))
    def ci(v): return [round(statistics.quantiles(v, n=40)[0], 3), round(statistics.quantiles(v, n=40)[38], 3)]  # ~2.5/97.5
    out = {
        "n_runs": N_RUNS, "n_cases": len(ids), "bootstrap_B": B,
        "micro_F1": round(micro_pt, 3), "micro_F1_CI95": ci(mic),
        "macro_F1": round(macro_pt, 3), "macro_F1_CI95": ci(mac),
        "principal_location_recall": round(sum(ploc.values()) / len(ploc), 3),
        "principal_location_hits": f"{sum(ploc.values())}/{len(ploc)}",
        "date_coverage_mean": round(statistics.mean(dcov.values()), 3),
        "date_coverage_by_case": {k: round(v, 2) for k, v in dcov.items()},
        "examples": examples[:6],
    }
    (Path(__file__).parent / "experiment_ci.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
