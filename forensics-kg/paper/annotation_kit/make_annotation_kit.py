#!/usr/bin/env python3
"""Generate a BLIND second-annotator kit for the 28 documented cases.

Each output file gives the case title, source link, and factual narrative, plus an
EMPTY `annotation` template. The first annotator's gold is NOT included, so the
second annotator labels independently. After it is filled, compute_agreement.py
computes inter-annotator agreement (pairwise F1 + Cohen's kappa) and an adjudicated
gold. Run from anywhere: writes ./cases/<case_id>.json next to this script.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SRC = ROOT / "data" / "real_cases"

FILL = ("Label INDEPENDENTLY from the narrative (consult `source` only to "
        "disambiguate). Do NOT view any existing gold. "
        "persons = [[\"Full Name\",\"role\"], ...] with role in "
        "{victim, accused, witness, officer, forensic_analyst}; "
        "locations = [\"single principal scene\"]; "
        "weapons = [\"weapon or method as a noun\"]; "
        "crime_types = [\"canonical type\", ...] (e.g. murder, rape, gang rape, "
        "kidnapping, robbery, hit-and-run); "
        "time_events = [\"documented date(s)\"] as stated (full date or year). "
        "Record only facts the case asserts, not inferences.")


def main():
    out_dir = HERE / "cases"
    out_dir.mkdir(exist_ok=True)
    cases = sorted(SRC.glob("*.json"))
    for f in cases:
        d = json.loads(f.read_text())
        tmpl = {
            "case_id": d["case_id"],
            "title": d.get("title", d["case_id"]),
            "source": d.get("source", ""),
            "narrative": d["narrative"],
            "_fill_below": FILL,
            "annotation": {
                "persons": [],
                "locations": [],
                "weapons": [],
                "crime_types": [],
                "time_events": [],
            },
        }
        (out_dir / f.name).write_text(json.dumps(tmpl, indent=2, ensure_ascii=False))
    print(f"wrote {len(cases)} blind templates to {out_dir}")


if __name__ == "__main__":
    main()
