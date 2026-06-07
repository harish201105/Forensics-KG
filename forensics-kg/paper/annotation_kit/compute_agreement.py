#!/usr/bin/env python3
"""Inter-annotator agreement between annotator-1 (existing gold in data/real_cases)
and annotator-2 (the filled blind kit in ./cases). Produces:

  * pairwise F1 (positive specific agreement) for the open-set categories
    (Person/Location/Weapon mentions, CrimeType set, TimeEvent dates) -- the
    standard metric for entity annotation, where Cohen's kappa is degenerate
    because there is no closed "true-negative" universe;
  * Cohen's kappa for the genuinely CATEGORICAL decisions: the role label of
    each shared Person, and each case's primary crime type;
  * an adjudication report listing every disagreement, so a final gold can be
    settled.

Usage:
  python3 compute_agreement.py            # annotator-2 (cases/annotated_cases) vs annotator-1
  python3 compute_agreement.py expert     # a third pass in cases/expert/ vs annotator-1
  python3 compute_agreement.py --selftest # annotator-1 vs itself (=> 1.0)
Writes agreement_report.json.
"""
import json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.services.evaluation.evaluator import _compute_metrics
from app.services.evaluation.real_case_evaluator import _canon_crimes, _date_match

ROLE_CANON = ("victim", "accused", "witness", "officer", "forensic_analyst")


def role_norm(r):
    r = (r or "").lower()
    for c in ROLE_CANON:
        if c.split("_")[0] in r:
            return c
    return r.strip() or "unknown"


def primary_crime(crimes):
    cs = _canon_crimes(" ".join(crimes or []))
    if "murder" in cs:
        return "murder"
    return sorted(cs)[0] if cs else "none"


def cohen_kappa(pairs):
    if not pairs:
        return None
    n = len(pairs)
    labels = set(a for a, _ in pairs) | set(b for _, b in pairs)
    po = sum(1 for a, b in pairs if a == b) / n
    ca, cb = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return round((po - pe) / (1 - pe), 3) if (1 - pe) > 1e-9 else 1.0


def f1_from(g, p):
    m = _compute_metrics("x", g, p)
    return m.true_positives, len(p) - m.true_positives, len(g) - m.true_positives, m.matched_pairs


def date_f1(g, p):
    used, tp = set(), 0
    for gd in g:
        for k, pd in enumerate(p):
            if k in used:
                continue
            if _date_match(gd, pd):
                used.add(k); tp += 1; break
    return tp, len(p) - tp, len(g) - tp


def load(argdir, selftest):
    a1 = {f.stem: json.loads(f.read_text())["gold"] for f in sorted((ROOT / "data/real_cases").glob("*.json"))}
    if selftest:
        return a1, a1
    # Optional CLI dir (e.g. a forensic-expert pass in cases/expert/); else
    # default to cases/annotated_cases/, else the cases/ templates.
    if argdir:
        cases_dir = Path(argdir)
        if not cases_dir.is_absolute():
            cases_dir = HERE / "cases" / argdir
    else:
        cases_dir = HERE / "cases" / "annotated_cases"
        if not cases_dir.exists() or not list(cases_dir.glob("*.json")):
            cases_dir = HERE / "cases"
    a2 = {}
    for f in sorted(cases_dir.glob("*.json")):
        a2[f.stem] = json.loads(f.read_text())["annotation"]
    return a1, a2


def main():
    selftest = "--selftest" in sys.argv
    argdir = next((a for a in sys.argv[1:] if not a.startswith("-")), None)
    a1, a2 = load(argdir, selftest)
    filled = sum(1 for v in a2.values() if any(v.get(k) for k in ("persons", "locations", "weapons", "crime_types", "time_events")))
    if filled == 0:
        print(f"Annotator-2 kit is empty ({len(a2)} templates in ./cases). "
              "Fill the `annotation` blocks, then re-run. (Use --selftest to validate the scorer.)")
        return

    agg = {t: [0, 0, 0] for t in ("Person", "Location", "Weapon", "CrimeType", "TimeEvent")}
    role_pairs, crime_pairs, disagreements = [], [], []

    for cid in a1:
        g, p = a1[cid], a2.get(cid, {})
        # Person names + role kappa
        gnames = [x[0] for x in g.get("persons", [])]
        pnames = [x[0] for x in p.get("persons", [])]
        grole = {x[0]: role_norm(x[1]) for x in g.get("persons", [])}
        prole = {x[0]: role_norm(x[1]) for x in p.get("persons", [])}
        tp, fp, fn, matched = f1_from(gnames, pnames)
        agg["Person"] = [agg["Person"][i] + v for i, v in enumerate((tp, fp, fn))]
        for mp in matched:
            role_pairs.append((grole.get(mp.gold, "unknown"), prole.get(mp.extracted, "unknown")))
            if grole.get(mp.gold) != prole.get(mp.extracted):
                disagreements.append(f"{cid}: role {mp.gold!r} A1={grole.get(mp.gold)} vs A2={prole.get(mp.extracted)}")
        for nm in set(gnames) - {mp.gold for mp in matched}:
            disagreements.append(f"{cid}: person only in A1: {nm!r}")
        for nm in set(pnames) - {mp.extracted for mp in matched}:
            disagreements.append(f"{cid}: person only in A2: {nm!r}")
        # Location / Weapon
        for cat, key in (("Location", "locations"), ("Weapon", "weapons")):
            tp, fp, fn, mtc = f1_from(g.get(key, []), p.get(key, []))
            agg[cat] = [agg[cat][i] + v for i, v in enumerate((tp, fp, fn))]
            matchedg = {m.gold for m in mtc}; matchedp = {m.extracted for m in mtc}
            for v in [x for x in g.get(key, []) if x not in matchedg]:
                disagreements.append(f"{cid}: {cat} only in A1: {v!r}")
            for v in [x for x in p.get(key, []) if x not in matchedp]:
                disagreements.append(f"{cid}: {cat} only in A2: {v!r}")
        # CrimeType (canonical set)
        gc, pc = _canon_crimes(" ".join(g.get("crime_types", []))), _canon_crimes(" ".join(p.get("crime_types", [])))
        agg["CrimeType"] = [agg["CrimeType"][0] + len(gc & pc), agg["CrimeType"][1] + len(pc - gc), agg["CrimeType"][2] + len(gc - pc)]
        for v in gc - pc:
            disagreements.append(f"{cid}: CrimeType only in A1: {v!r}")
        for v in pc - gc:
            disagreements.append(f"{cid}: CrimeType only in A2: {v!r}")
        crime_pairs.append((primary_crime(g.get("crime_types", [])), primary_crime(p.get("crime_types", []))))
        # TimeEvent (date-aware)
        tp, fp, fn = date_f1([str(x) for x in g.get("time_events", [])], [str(x) for x in p.get("time_events", [])])
        agg["TimeEvent"] = [agg["TimeEvent"][i] + v for i, v in enumerate((tp, fp, fn))]

    def f1(tp, fp, fn):
        return round(2 * tp / (2 * tp + fp + fn), 3) if (2 * tp + fp + fn) else 1.0

    out = {
        "n_cases": len(a1),
        "pairwise_F1": {t: f1(*agg[t]) for t in agg},
        "person_role_cohen_kappa": cohen_kappa(role_pairs),
        "primary_crime_cohen_kappa": cohen_kappa(crime_pairs),
        "n_role_pairs": len(role_pairs),
        "n_disagreements": len(disagreements),
        "disagreements": disagreements,
    }
    (HERE / "agreement_report.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"}, indent=2))
    print(f"\n{len(disagreements)} disagreements written to agreement_report.json")


if __name__ == "__main__":
    main()
