#!/usr/bin/env python3
"""Quantitative cross-case entity-resolution benchmark.

Extracts entities from the 10 documented cases, then runs the SAME resolution
logic the system uses (embedding blocking -> name/role/context decision) over all
cross-case pairs, and prints the accepted SAME_AS links plus the candidate-pool
size so the links can be labelled true/false for precision. For recall we also
enumerate an objective gold of "should-link" pairs (cross-case CrimeType-name and
city-level Location identities). Writes resolution_eval.json.
"""
import asyncio, json, sys, itertools
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings
from app.services.extraction.openai_client import OpenAIClient
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.text_extractor import TextExtractor
from app.services.graph.entity_resolution import (
    _normalize, _name_score, _decide_match, _jaccard, RESOLVABLE, _GENERIC,
)

settings = get_settings()
NAME_TH, BLOCK_TH = 0.88, 0.78   # the system's defaults


def cos(a, b):
    import math
    s = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return s/(na*nb) if na and nb else 0.0


async def main():
    oc = OpenAIClient(settings)
    te = TextExtractor(oc, ForensicsOntologySchema(settings.ontology_path))
    cases = [json.loads(p.read_text()) for p in sorted((Path(settings.data_dir) / "real_cases").glob("*.json"))]

    # 1. extract entities per case
    items = []   # each: {case, label, val, norm, role, ctx(set), text}
    for c in cases:
        ents = (await te.extract(c["narrative"], "fir")).get("entities", [])
        # same-case context = normalized names of all resolvable siblings
        sib = set()
        rows = []
        for e in ents:
            lab = e.get("entity_type"); prop = RESOLVABLE.get(lab)
            if not prop:
                continue
            val = (e.get("properties", {}) or {}).get(prop, "")
            norm = _normalize(lab, val)
            if not norm or norm in _GENERIC:
                continue
            role = ((e.get("properties", {}) or {}).get("role") or "").lower()
            rows.append({"case": c["case_id"], "label": lab, "val": val,
                         "norm": norm, "role": role, "text": f"{lab}: {val}"})
            sib.add(norm)
        for r in rows:
            r["ctx"] = sib - {r["norm"]}
        items.extend(rows)
        print(f"  {c['case_id']}: {len(rows)} resolvable entities", flush=True)

    # 2. embed all entity texts
    vecs = await oc.embed([it["text"] for it in items])
    for it, v in zip(items, vecs):
        it["emb"] = v

    # 3. cross-case blocking + decision, per label
    accepted, all_candidates, candidates = [], [], 0
    n_entities = len(items)
    all_cross = sum(1 for a, b in itertools.combinations(range(n_entities), 2)
                    if items[a]["label"] == items[b]["label"]
                    and items[a]["case"] != items[b]["case"])
    by_label = {}
    for lab in RESOLVABLE:
        idx = [i for i, it in enumerate(items) if it["label"] == lab]
        by_label[lab] = {"nodes": len(idx), "accepted": 0}
        for a, b in itertools.combinations(idx, 2):
            ia, ib = items[a], items[b]
            if ia["case"] == ib["case"]:
                continue
            if cos(ia["emb"], ib["emb"]) < BLOCK_TH:
                continue
            candidates += 1
            ns, min_tok, _, conflict = _name_score(ia["norm"], ib["norm"])
            ctx = _jaccard(ia["ctx"], ib["ctx"])
            ok, conf, basis = _decide_match(ns, min_tok, conflict, ia["role"], ib["role"], ctx, NAME_TH)
            rec = {"label": lab, "a": f"{ia['val']} [{ia['case']}]",
                   "b": f"{ib['val']} [{ib['case']}]",
                   "cos": round(cos(ia["emb"], ib["emb"]), 2),
                   "name_score": round(ns, 2), "accepted": bool(ok),
                   "conf": round(conf, 2), "basis": basis}
            all_candidates.append(rec)
            if ok:
                by_label[lab]["accepted"] += 1
                accepted.append(rec)

    # 4. objective recall gold: cross-case CrimeType-name identities + city-Location
    def gold_pairs(label, key, normfn):
        # cases grouped by a canonical value -> all cross-case pairs are "should link"
        buckets = {}
        for c in cases:
            for v in c["gold"].get(key, []) or []:
                cv = normfn(v)
                if cv:
                    buckets.setdefault(cv, set()).add(c["case_id"])
        pairs = set()
        for cv, cs in buckets.items():
            for x, y in itertools.combinations(sorted(cs), 2):
                pairs.add((cv, x, y))
        return pairs

    def crime_norm(v): return v.strip().lower()
    def city_norm(v):
        v = v.lower()
        for city in ("delhi", "mumbai", "kozhikode", "ghaziabad", "noida", "kolkata"):
            if city in v:
                return city
        return ""
    crime_gold = gold_pairs("CrimeType", "crime_types", crime_norm)
    city_gold = gold_pairs("Location", "locations", city_norm)

    out = {"n_entities": len(items), "all_cross_case_pairs": all_cross,
           "candidate_pairs": candidates, "accepted_links": len(accepted),
           "by_label": by_label, "accepted": accepted,
           "all_candidates": all_candidates,
           "objective_gold": {"crimetype_pairs": len(crime_gold),
                              "city_location_pairs": len(city_gold),
                              "crime_examples": sorted(crime_gold)[:8],
                              "city_examples": sorted(city_gold)[:8]}}
    (Path(__file__).parent / "resolution_eval.json").write_text(json.dumps(out, indent=2))
    print(f"entities={len(items)} all_cross_pairs={all_cross} candidates={candidates} accepted={len(accepted)}")
    print("\n--- ALL candidate pairs (post-block) ---")
    for i, c in enumerate(all_candidates):
        print(f"{i+1:2}. [{c['label']:9}] {'ACCEPT' if c['accepted'] else 'reject'} "
              f"cos={c['cos']} name={c['name_score']} | {c['a']} == {c['b']}")


if __name__ == "__main__":
    asyncio.run(main())
