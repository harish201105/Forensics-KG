#!/usr/bin/env python3
"""Containment-aware principal-location recall, alongside the strict metric.

The strict per-mention metric penalises the system for extracting a MORE specific
location than the gold (``St Pius X Convent, Kottayam'' vs gold ``Kottayam''). The
human inter-annotator study (sec:iaa) shows two annotators name the same place but
at different granularity. This script reports a complementary, granularity-tolerant
``place-level'' recall: a gold principal location is recovered if some extracted
location's place tokens are a superset OR subset of the gold's (i.e. same place,
any granularity), or they fuzzy-match >=0.75. Genuinely different places (Chennai
vs Kodaikanal; Chhaparvad vs Randhikpur) are NOT credited. One run. Writes
location_analysis.json.
"""
import asyncio, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings
from app.services.extraction.openai_client import OpenAIClient
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.text_extractor import TextExtractor
from app.services.evaluation.evaluator import _fuzzy_match

settings = get_settings()
STOP = {"the", "of", "a", "an", "near", "at", "in", "district", "region", "new", "and"}


def toks(s):
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return {t for t in s.split() if len(t) > 2 and t not in STOP}


def place_match(gold, ext):
    g = toks(gold)
    if not g:
        return False
    for e in ext:
        et = toks(e)
        if not et:
            continue
        if g <= et or et <= g:          # same place, finer/coarser granularity
            return True
        if _fuzzy_match(gold, e) >= 0.75:
            return True
    return False


def strict_match(gold, ext):
    return any(_fuzzy_match(gold, e) >= 0.75 for e in ext)


async def main():
    oc = OpenAIClient(settings)
    te = TextExtractor(oc, ForensicsOntologySchema(settings.ontology_path))
    cases = [json.loads(p.read_text()) for p in sorted((Path(settings.data_dir) / "real_cases").glob("*.json"))]
    rows, strict_hits, place_hits, flipped = [], 0, 0, []
    for c in cases:
        ents = (await te.extract(c["narrative"], "fir")).get("entities", [])
        ext = [e["properties"].get("name", "") for e in ents if e.get("entity_type") == "Location"]
        gold = (c["gold"].get("locations") or [""])[0]
        s = strict_match(gold, ext)
        p = place_match(gold, ext)
        strict_hits += s
        place_hits += p
        if p and not s:
            flipped.append({"case": c["case_id"], "gold": gold, "extracted": ext})
        rows.append({"case": c["case_id"], "gold": gold, "strict": s, "place": p, "extracted": ext})
        print(f"  {c['case_id']:<22} strict={int(s)} place={int(p)}", flush=True)
    n = len(cases)
    out = {"n_cases": n,
           "strict_principal_recall": round(strict_hits / n, 3),
           "containment_principal_recall": round(place_hits / n, 3),
           "strict_hits": f"{strict_hits}/{n}", "place_hits": f"{place_hits}/{n}",
           "flipped_miss_to_hit": flipped, "rows": rows}
    (Path(__file__).parent / "location_analysis.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"\nstrict principal-loc recall   = {out['strict_principal_recall']} ({out['strict_hits']})")
    print(f"containment principal-loc recall = {out['containment_principal_recall']} ({out['place_hits']})")
    print(f"cases credited by granularity tolerance: {len(flipped)}")


if __name__ == "__main__":
    asyncio.run(main())
