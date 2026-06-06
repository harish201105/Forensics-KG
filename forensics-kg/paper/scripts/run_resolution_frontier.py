#!/usr/bin/env python3
"""Calibrated tiered cross-case resolver: precision / recall / review-load frontier.

Offline analysis of the labelled candidate pool in resolution_eval.json (no API).
The deployed resolver uses ONE conservative operating point (auto-accept only when a
name match is corroborated by context/role) -> precision 0.96, recall 0.15. Here we
sweep the decision and report the full frontier, and a TIERED scheme
(auto-accept / human-review / auto-reject) that recovers far more true links at a
quantified human-review cost. We also split CATEGORY-type nodes (CrimeType: "Murder"
== "Murder" is a correct category merge) from INSTANCE-type nodes (Person/Location/
Weapon: identical names can be collisions, e.g. two different "Ajay Kumar"), because
the safe operating point differs by type. Writes resolution_frontier.json.
"""
import json, re
from pathlib import Path

HERE = Path(__file__).resolve().parent
d = json.load(open(HERE / "resolution_eval.json"))
C = d["all_candidates"]

# Known instance-level FALSE merge at identical name (verified from sources):
# a witness "Ajay Kumar" (Nitish Katara) vs an accused of the same common name
# (Soumya Vishwanathan). Generic weapon-type identicals ("firearm"=="firearm")
# are type-level, not the same physical weapon -> not instance identity.
CATEGORY = {"CrimeType"}


def norm(s):
    return re.sub(r"\s*\[.*?\]", "", s).strip().lower()


def is_collision(c):
    # Same should-link labelling as the resolution benchmark (Sec. RQ4): the only
    # verified instance-level false merge at an identical name is the common-name
    # person collision (a witness "Ajay Kumar" vs an accused of the same name).
    a, b = norm(c["a"]), norm(c["b"])
    if c["label"] == "Person" and "ajay kumar" in a and "ajay kumar" in b:
        return True
    return False


def should_link(c):
    """Correct cross-case identity (objective rule)."""
    if is_collision(c):
        return False
    if c["label"] == "CrimeType":
        return c["name_score"] >= 0.90
    if c["label"] == "Location":
        return c["name_score"] >= 0.90
    return norm(c["a"]) == norm(c["b"])      # Person/Weapon/Vehicle: same name


for c in C:
    c["gold"] = should_link(c)
    c["cat"] = "category" if c["label"] in CATEGORY else "instance"

TOTAL_SL = sum(c["gold"] for c in C)          # should-link pairs in the pool
N_CASES = 28


def metrics(accepted):
    tp = sum(1 for c in accepted if c["gold"])
    fp = len(accepted) - tp
    P = tp / len(accepted) if accepted else 1.0
    R = tp / TOTAL_SL if TOTAL_SL else 0.0
    return round(P, 3), round(R, 3), tp, fp


# ---- frontier: sweep an acceptance threshold on name_score ----
thresholds = sorted({round(c["name_score"], 2) for c in C}, reverse=True)
frontier = []
for t in thresholds:
    acc = [c for c in C if c["name_score"] >= t]
    P, R, tp, fp = metrics(acc)
    frontier.append({"name_thr": t, "accepted": len(acc), "P": P, "R": R, "tp": tp, "fp": fp})

# ---- the three operating points ----
# 1) Conservative (the deployed system): exactly what it auto-accepted
cons = [c for c in C if c["accepted"]]
# 2) Type-aware auto-merge: auto-accept identical CATEGORY nodes (safe) + the
#    conservative instance accepts; route uncorroborated identical INSTANCE names to review.
auto = [c for c in C if (c["cat"] == "category" and c["name_score"] >= 0.99) or c["accepted"]]
review = [c for c in C if c["cat"] == "instance" and c["name_score"] >= 0.99 and not c["accepted"]]
# 3) High-recall (accept all near-identical names, no review) -- shows the precision cost
hi = [c for c in C if c["name_score"] >= 0.95]


def line(name, acc, rev=None):
    P, R, tp, fp = metrics(acc)
    s = f"{name:<34} P={P:.2f} R={R:.2f}  (TP={tp} FP={fp} accepted={len(acc)})"
    if rev is not None:
        # recall recoverable if the review tier's true links are confirmed
        rtp = sum(1 for c in rev if c["gold"]); rfp = len(rev) - rtp
        R2 = (tp + rtp) / TOTAL_SL
        s += f"  +review {len(rev)} pairs ({len(rev)/N_CASES:.1f}/case: {rtp} true,{rfp} collision) -> recall {R2:.2f}"
    return s


def split_by_task(accepted):
    res = {}
    for grp in ("instance", "category"):
        pool = [x for x in C if x["cat"] == grp]
        tot = sum(x["gold"] for x in pool)
        acc = [x for x in accepted if x["cat"] == grp]
        tp = sum(x["gold"] for x in acc); fp = len(acc) - tp
        res[grp] = {"P": round(tp / len(acc), 2) if acc else 1.0,
                    "R": round(tp / tot, 2) if tot else 0.0, "tp": tp, "fp": fp, "pool_sl": tot}
    return res


out = {
    "pool_size": len(C), "should_link_in_pool": TOTAL_SL, "n_cases": N_CASES,
    "by_task_conservative": split_by_task([c for c in C if c["accepted"]]),
    "by_task_relaxed_nameidentity": split_by_task([c for c in C if c["name_score"] >= 0.95]),
    "category_should_link": sum(c["gold"] for c in C if c["cat"] == "category"),
    "instance_should_link": sum(c["gold"] for c in C if c["cat"] == "instance"),
    "conservative": dict(zip(("P", "R", "tp", "fp"), metrics(cons))),
    "type_aware_auto": dict(zip(("P", "R", "tp", "fp"), metrics(auto))),
    "type_aware_review_pairs": len(review),
    "type_aware_review_per_case": round(len(review) / N_CASES, 2),
    "type_aware_review_true": sum(1 for c in review if c["gold"]),
    "type_aware_review_collision": sum(1 for c in review if not c["gold"]),
    "high_recall_norescue": dict(zip(("P", "R", "tp", "fp"), metrics(hi))),
    "frontier": frontier,
}
(HERE / "resolution_frontier.json").write_text(json.dumps(out, indent=2))

print(f"pool={len(C)}  should-link in pool={TOTAL_SL}  (category={out['category_should_link']}, instance={out['instance_should_link']})\n")
print(line("1. Conservative (deployed)", cons))
print(line("2. Type-aware auto + review", auto, review))
print(line("3. High-recall (name>=0.95, no review)", hi))
print("\n--- frontier (name_thr: P / R / accepted) ---")
for f in frontier:
    print(f"  thr={f['name_thr']:.2f}  P={f['P']:.2f} R={f['R']:.2f}  accepted={f['accepted']}")
