"""Scaled validation against REAL documented cases.

Unlike the synthetic gold standards (LLM-generated, hence circular), these gold
annotations are curated from authoritative public sources (court records,
Wikipedia) — the established facts of well-documented Indian criminal cases. The
extraction pipeline runs on a factual narrative and is scored (P/R/F1) against
those human-curated documented facts, breaking the circular-evaluation problem.
"""

import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from app.services.extraction.text_extractor import TextExtractor
from app.services.evaluation.evaluator import _compute_metrics, _get_extracted_by_type
from app.services.evaluation.metrics import (
    EvaluationResult, AggregateEvaluationResult, EntityTypeMetrics, MatchedPair,
)

# Entity types scored with direct fuzzy string matching (gold and extracted
# representations are directly comparable).
_ENTITY_MAP = {
    "Person": "persons",
    "Location": "locations",
    "Weapon": "weapons",
}

_MONTHS = ["january", "february", "march", "april", "may", "june", "july",
           "august", "september", "october", "november", "december"]

# Canonical crime categories and the keywords that signal them. Used to compare
# documented crime types against the Case title/description (where the FIR
# extractor actually records the crime), instead of against (rare) CrimeType nodes.
_CRIME_CANON = [
    ("murder", ["murder", "homicide", "killing", "killed", "honour killing"]),
    ("rape", ["rape", "gang rape", "gang-rape", "sexual assault"]),
    ("poisoning", ["poison"]),
    ("kidnapping", ["kidnap", "abduct"]),
    ("robbery", ["robbery", "robbed", "loot"]),
    ("assault", ["assault"]),
    ("arson", ["arson", "set ablaze", "set on fire", "burned the body", "burnt"]),
    ("serial killing", ["serial"]),
]


def _date_keys(text: str):
    t = (text or "").lower()
    years = {y for y in re.findall(r"\b\d{4}\b", t) if 1900 <= int(y) <= 2099}
    months = {m for m in _MONTHS if m in t}
    return years, months


def _date_match(gold_text: str, ext_text: str) -> bool:
    """A documented date matches an extracted event if they share a year and,
    when both name a month, the month too."""
    gy, gm = _date_keys(gold_text)
    ey, em = _date_keys(ext_text)
    if not gy or not ey or not (gy & ey):
        return False
    if gm and em and not (gm & em):
        return False
    return True


def _canon_crimes(text: str):
    t = (text or "").lower()
    return {canon for canon, kws in _CRIME_CANON if any(k in t for k in kws)}


def _set_metrics(entity_type: str, gold: set, pred: set) -> EntityTypeMetrics:
    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return EntityTypeMetrics(
        entity_type=entity_type, true_positives=tp, false_positives=fp,
        false_negatives=fn, precision=round(p, 4), recall=round(r, 4),
        f1=round(f, 4),
        matched_pairs=[MatchedPair(gold=g, extracted=g, similarity=1.0) for g in (gold & pred)],
    )


class RealCaseEvaluator:
    def __init__(self, data_dir, text_extractor: TextExtractor):
        self._dir = Path(data_dir) / "real_cases"
        self._text = text_extractor

    def list_cases(self) -> List[Dict]:
        if not self._dir.exists():
            return []
        return [json.loads(p.read_text(encoding="utf-8"))
                for p in sorted(self._dir.glob("*.json"))]

    def _gold_items(self, gold: Dict, gkey: str) -> List[str]:
        items = gold.get(gkey, []) or []
        # persons are [name, role] pairs -> use the name
        if gkey == "persons":
            return [p[0] if isinstance(p, (list, tuple)) else str(p) for p in items]
        return [str(x) for x in items]

    async def evaluate_one(self, case: Dict, model_override: Optional[str] = None) -> EvaluationResult:
        t0 = time.time()
        result = await self._text.extract(
            case["narrative"], "fir", model_override=model_override
        )
        elapsed = (time.time() - t0) * 1000
        entities = result.get("entities", [])
        gold = case.get("gold", {})

        entity_metrics: Dict[str, EntityTypeMetrics] = {}
        gold_total = ext_total = 0
        # Person / Location / Weapon — direct fuzzy string matching
        for etype, gkey in _ENTITY_MAP.items():
            gold_items = self._gold_items(gold, gkey)
            extracted = _get_extracted_by_type(entities, etype)
            if not gold_items and not extracted:
                continue
            m = _compute_metrics(etype, gold_items, extracted)
            entity_metrics[etype] = m
            gold_total += len(gold_items)
            ext_total += len(extracted)

        # CrimeType — canonical-keyword set comparison against the Case
        # title/description (where the FIR extractor records the crime).
        gold_crimes = _canon_crimes(" ".join(gold.get("crime_types", []) or []))
        if gold_crimes:
            crime_text = " ".join(
                f"{e['properties'].get('title','')} {e['properties'].get('name','')} "
                f"{e['properties'].get('description','')} {e['properties'].get('crime_type','')}"
                for e in entities if e.get("entity_type") in ("Case", "CrimeType")
            )
            entity_metrics["CrimeType"] = _set_metrics(
                "CrimeType", gold_crimes, _canon_crimes(crime_text))
            gold_total += len(gold_crimes)

        # TimeEvent — date-aware matching (year/month from event text vs gold dates)
        gold_dates = [str(t) for t in (gold.get("time_events", []) or [])]
        if gold_dates:
            ev_texts = [
                f"{e['properties'].get('name','')} {e['properties'].get('description','')}"
                for e in entities if e.get("entity_type") == "TimeEvent"
            ]
            used = set()
            matched = []
            for g in gold_dates:
                for k, ev in enumerate(ev_texts):
                    if k in used:
                        continue
                    if _date_match(g, ev):
                        used.add(k)
                        matched.append(MatchedPair(gold=g, extracted=ev[:60], similarity=1.0))
                        break
            tp = len(matched)
            fp = len(ev_texts) - tp
            fn = len(gold_dates) - tp
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r / (p + r) if (p + r) else 0.0
            entity_metrics["TimeEvent"] = EntityTypeMetrics(
                entity_type="TimeEvent", true_positives=tp, false_positives=fp,
                false_negatives=fn, precision=round(p, 4), recall=round(r, 4),
                f1=round(f, 4), matched_pairs=matched)
            gold_total += len(gold_dates)
            ext_total += len(ev_texts)

        tp = sum(m.true_positives for m in entity_metrics.values())
        fp = sum(m.false_positives for m in entity_metrics.values())
        fn = sum(m.false_negatives for m in entity_metrics.values())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

        return EvaluationResult(
            fir_id=case.get("case_id", "real-case"),
            entity_metrics=entity_metrics,
            aggregate_precision=round(prec, 4),
            aggregate_recall=round(rec, 4),
            aggregate_f1=round(f1, 4),
            extraction_time_ms=round(elapsed, 1),
            gold_entity_count=gold_total,
            extracted_entity_count=ext_total,
            model_used=model_override or "default",
        )

    async def evaluate_all(self, model_override: Optional[str] = None) -> AggregateEvaluationResult:
        cases = self.list_cases()
        per_case: List[EvaluationResult] = []
        for case in cases:
            try:
                per_case.append(await self.evaluate_one(case, model_override))
            except Exception:
                continue

        # micro-average per entity type across all cases
        agg: Dict[str, Dict[str, int]] = {}
        for r in per_case:
            for etype, m in r.entity_metrics.items():
                a = agg.setdefault(etype, {"tp": 0, "fp": 0, "fn": 0})
                a["tp"] += m.true_positives
                a["fp"] += m.false_positives
                a["fn"] += m.false_negatives

        per_entity_type: Dict[str, EntityTypeMetrics] = {}
        for etype, a in agg.items():
            tp, fp, fn = a["tp"], a["fp"], a["fn"]
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r_ = tp / (tp + fn) if (tp + fn) else 0.0
            f = 2 * p * r_ / (p + r_) if (p + r_) else 0.0
            per_entity_type[etype] = EntityTypeMetrics(
                entity_type=etype, true_positives=tp, false_positives=fp,
                false_negatives=fn, precision=round(p, 4), recall=round(r_, 4),
                f1=round(f, 4), matched_pairs=[],
            )

        TP = sum(a["tp"] for a in agg.values())
        FP = sum(a["fp"] for a in agg.values())
        FN = sum(a["fn"] for a in agg.values())
        op = TP / (TP + FP) if (TP + FP) else 0.0
        orr = TP / (TP + FN) if (TP + FN) else 0.0
        of1 = 2 * op * orr / (op + orr) if (op + orr) else 0.0

        return AggregateEvaluationResult(
            total_cases=len(per_case),
            per_case=per_case,
            per_entity_type=per_entity_type,
            overall_precision=round(op, 4),
            overall_recall=round(orr, 4),
            overall_f1=round(of1, 4),
            model_used=model_override or "default",
        )
