"""Core evaluation logic: compare extracted entities against gold standard."""

import json
import re
import time
from abc import ABC, abstractmethod
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from app.services.extraction.text_extractor import TextExtractor
from app.services.evaluation.metrics import (
    EntityTypeMetrics,
    EvaluationResult,
    AggregateEvaluationResult,
    MatchedPair,
)


def _normalize(text: str) -> str:
    """Normalize text for fuzzy matching."""
    text = text.lower().strip()
    text = re.sub(r"\b(mr|mrs|ms|dr|prof|sir|smt|justice|hon'ble|shri)\.?\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _fuzzy_match(a: str, b: str, threshold: float = 0.75) -> float:
    """Return similarity ratio between two strings."""
    na, nb = _normalize(a), _normalize(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _compute_metrics(
    entity_type: str, gold_items: List[str], extracted_items: List[str], threshold: float = 0.75
) -> EntityTypeMetrics:
    """Match gold items to extracted items greedily, compute P/R/F1."""
    matched_pairs: List[MatchedPair] = []
    used_extracted = set()

    for g in gold_items:
        best_score = 0.0
        best_idx = -1
        for j, e in enumerate(extracted_items):
            if j in used_extracted:
                continue
            score = _fuzzy_match(g, e, threshold)
            if score > best_score:
                best_score = score
                best_idx = j
        if best_score >= threshold and best_idx >= 0:
            used_extracted.add(best_idx)
            matched_pairs.append(MatchedPair(
                gold=g, extracted=extracted_items[best_idx], similarity=round(best_score, 3)
            ))

    tp = len(matched_pairs)
    fp = len(extracted_items) - tp
    fn = len(gold_items) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return EntityTypeMetrics(
        entity_type=entity_type,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        matched_pairs=matched_pairs,
    )


def _get_extracted_by_type(
    entities: List[Dict], entity_type: str, name_key: str = "name"
) -> List[str]:
    """Get name/description strings from extracted entities of a given type."""
    items = []
    for e in entities:
        if e.get("entity_type") == entity_type:
            props = e.get("properties", {})
            name = props.get(name_key) or props.get("description") or props.get("title") or ""
            if name:
                items.append(str(name))
    return items


# --- Gold Standard Adapters ---

class GoldStandardAdapter(ABC):
    """Abstract adapter for extracting ground truth from different gold formats."""

    @abstractmethod
    def extract_gold_items(self, gold: Dict) -> Dict[str, List[str]]:
        """Return {entity_type: [gold_item_strings]} from gold standard JSON."""

    @abstractmethod
    def get_source_type(self) -> str:
        """Return the source_type string for extraction."""

    @abstractmethod
    def get_text_dir(self) -> str:
        """Return the subdirectory name for text files."""

    @abstractmethod
    def get_gold_dir(self) -> str:
        """Return the subdirectory name for gold files."""


class FIRGoldAdapter(GoldStandardAdapter):
    """Gold adapter for FIR reports (original format)."""

    def get_source_type(self) -> str:
        return "fir"

    def get_text_dir(self) -> str:
        return "fir_text"

    def get_gold_dir(self) -> str:
        return "gold"

    def extract_gold_items(self, gold: Dict) -> Dict[str, List[str]]:
        return {
            "Person": self._extract_persons(gold),
            "Location": self._extract_locations(gold),
            "Weapon": self._extract_weapons(gold),
            "Vehicle": self._extract_vehicles(gold),
            "TimeEvent": self._extract_timeline(gold),
            "Evidence": self._extract_evidence(gold),
        }

    def _extract_persons(self, gold: Dict) -> List[str]:
        persons = []
        if gold.get("complainant_name"):
            persons.append(gold["complainant_name"])
        if gold.get("victim_name"):
            persons.append(gold["victim_name"])
        suspect = gold.get("suspect_description", "")
        if suspect:
            match = re.match(r"^(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s*(.+?),", suspect)
            if match:
                persons.append(match.group(1).strip())
            else:
                words = suspect.split()
                if len(words) >= 2:
                    persons.append(" ".join(words[:2]))
        for ws in gold.get("witness_statements", []):
            match = re.match(r"^(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s*(.+?),", ws)
            if match:
                persons.append(match.group(1).strip())
        return persons

    def _extract_locations(self, gold: Dict) -> List[str]:
        locations = []
        loc_text = gold.get("location_details", "")
        if loc_text:
            parts = re.split(r"[.;]", loc_text)
            for part in parts:
                part = part.strip()
                if part and len(part) > 5:
                    if ":" in part:
                        part = part.split(":", 1)[1].strip()
                    locations.append(part)
        if gold.get("police_station"):
            locations.append(gold["police_station"])
        return locations

    def _extract_weapons(self, gold: Dict) -> List[str]:
        weapon = gold.get("weapon_details", "")
        return [weapon.strip()] if weapon and weapon.strip() else []

    def _extract_vehicles(self, gold: Dict) -> List[str]:
        vehicle = gold.get("vehicle_details", "")
        return [vehicle.strip()] if vehicle and vehicle.strip() else []

    def _extract_timeline(self, gold: Dict) -> List[str]:
        return [te["description"] for te in gold.get("timeline_events", []) if te.get("description")]

    def _extract_evidence(self, gold: Dict) -> List[str]:
        items = []
        if gold.get("evidence_description"):
            items.append(gold["evidence_description"])
        if gold.get("bloodstain_evidence"):
            items.append(gold["bloodstain_evidence"])
        return items


class CourtJudgmentGoldAdapter(GoldStandardAdapter):
    """Gold adapter for court judgment gold standards."""

    def get_source_type(self) -> str:
        return "court_judgment"

    def get_text_dir(self) -> str:
        return "court_judgments"

    def get_gold_dir(self) -> str:
        return "gold_court"

    def extract_gold_items(self, gold: Dict) -> Dict[str, List[str]]:
        persons = []
        persons.extend(gold.get("judge_names", []))
        persons.extend(gold.get("accused_names", []))
        persons.extend(gold.get("victim_names", []))
        persons.extend(gold.get("witness_names", []))
        persons.extend(gold.get("lawyer_names", []))

        locations = gold.get("locations", [])
        if gold.get("location_details"):
            locations.append(gold["location_details"])

        evidence = gold.get("evidence_items", [])
        timeline = [te["description"] for te in gold.get("timeline_events", []) if te.get("description")]

        legal_sections = [
            f"{s['section']} {s['act']}" for s in gold.get("legal_sections", [])
        ]

        weapons = [gold["weapon_details"]] if gold.get("weapon_details") else []
        vehicles = [gold["vehicle_details"]] if gold.get("vehicle_details") else []

        verdict = [gold["verdict"]] if gold.get("verdict") else []

        return {
            "Person": persons,
            "Location": locations,
            "Evidence": evidence,
            "TimeEvent": timeline,
            "LegalSection": legal_sections,
            "Weapon": weapons,
            "Vehicle": vehicles,
            "Verdict": verdict,
        }


class PostMortemGoldAdapter(GoldStandardAdapter):
    """Gold adapter for post-mortem reports."""

    def get_source_type(self) -> str:
        return "postmortem"

    def get_text_dir(self) -> str:
        return "postmortem"

    def get_gold_dir(self) -> str:
        return "gold_postmortem"

    def extract_gold_items(self, gold: Dict) -> Dict[str, List[str]]:
        persons = []
        if gold.get("deceased_name"):
            persons.append(gold["deceased_name"])
        if gold.get("examining_doctor"):
            persons.append(gold["examining_doctor"])

        injuries = [
            f"{inj['type']} at {inj['location']}" for inj in gold.get("injuries", [])
        ]
        tox = [t["substance"] for t in gold.get("toxicology_results", [])]
        organs = [o["organ"] for o in gold.get("organ_findings", [])]
        cod = [gold["cause_of_death"]] if gold.get("cause_of_death") else []

        return {
            "Person": persons,
            "InjuryPattern": injuries,
            "ToxicologyResult": tox,
            "OrganFinding": organs,
            "CauseOfDeath": cod,
        }


class LabReportGoldAdapter(GoldStandardAdapter):
    """Gold adapter for forensic lab reports."""

    def get_source_type(self) -> str:
        return "lab_report"

    def get_text_dir(self) -> str:
        return "lab_reports"

    def get_gold_dir(self) -> str:
        return "gold_lab"

    def extract_gold_items(self, gold: Dict) -> Dict[str, List[str]]:
        persons = []
        if gold.get("analyst_name"):
            persons.append(gold["analyst_name"])
        if gold.get("requesting_officer"):
            persons.append(gold["requesting_officer"])

        samples = [f"{s['sample_type']} from {s['source']}" for s in gold.get("samples", [])]
        tests = [f"{t['test_name']}: {t['result']}" for t in gold.get("tests_performed", [])]
        dna = [f"{d['profile_type']} {d['match_result']}" for d in gold.get("dna_profiles", [])]

        return {
            "Person": persons,
            "Sample": samples,
            "TestResult": tests,
            "DNAProfile": dna,
        }


class DepositionGoldAdapter(GoldStandardAdapter):
    """Gold adapter for witness depositions."""

    def get_source_type(self) -> str:
        return "witness_deposition"

    def get_text_dir(self) -> str:
        return "depositions"

    def get_gold_dir(self) -> str:
        return "gold_depositions"

    def extract_gold_items(self, gold: Dict) -> Dict[str, List[str]]:
        persons = gold.get("persons_mentioned", [])
        if gold.get("witness_name"):
            persons = [gold["witness_name"]] + persons
        if gold.get("recording_officer"):
            persons.append(gold["recording_officer"])

        statements = [c["claim"] for c in gold.get("key_claims", [])]
        locations = gold.get("locations_mentioned", [])
        timeline = [te["description"] for te in gold.get("timeline_events", []) if te.get("description")]

        return {
            "Person": persons,
            "Statement": statements,
            "Location": locations,
            "TimeEvent": timeline,
        }


class SOCOGoldAdapter(GoldStandardAdapter):
    """Gold adapter for SOCO reports."""

    def get_source_type(self) -> str:
        return "soco_report"

    def get_text_dir(self) -> str:
        return "soco"

    def get_gold_dir(self) -> str:
        return "gold_soco"

    def extract_gold_items(self, gold: Dict) -> Dict[str, List[str]]:
        persons = []
        if gold.get("scene_officer"):
            persons.append(gold["scene_officer"])
        for c in gold.get("chain_of_custody", []):
            if c.get("handler"):
                persons.append(c["handler"])

        evidence = [e["description"] for e in gold.get("evidence_collected", [])]
        custody = [
            f"{c['handler']} {c['action']}" for c in gold.get("chain_of_custody", [])
        ]

        return {
            "Person": persons,
            "PhysicalEvidence": evidence,
            "ChainOfCustody": custody,
        }


# Adapter registry
GOLD_ADAPTERS: Dict[str, GoldStandardAdapter] = {
    "fir": FIRGoldAdapter(),
    "court_judgment": CourtJudgmentGoldAdapter(),
    "postmortem": PostMortemGoldAdapter(),
    "lab_report": LabReportGoldAdapter(),
    "witness_deposition": DepositionGoldAdapter(),
    "soco_report": SOCOGoldAdapter(),
}


class ExtractionEvaluator:
    """Compare LLM-extracted entities against gold standard annotations."""

    def __init__(self, data_dir: Path, text_extractor: TextExtractor):
        self._data_dir = data_dir
        self._text_extractor = text_extractor

    def _get_adapter(self, source_type: str) -> GoldStandardAdapter:
        adapter = GOLD_ADAPTERS.get(source_type)
        if not adapter:
            logger.warning(f"No gold adapter for '{source_type}', falling back to 'fir'")
            adapter = GOLD_ADAPTERS["fir"]
        return adapter

    def _load_gold(self, doc_id: str, adapter: GoldStandardAdapter) -> Dict:
        path = self._data_dir / adapter.get_gold_dir() / f"{doc_id}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Gold standard not found: {path}. "
                f"Generate gold for '{doc_id}' first."
            )
        with open(path) as f:
            return json.load(f)

    def _load_text(self, doc_id: str, adapter: GoldStandardAdapter) -> str:
        text_dir = self._data_dir / adapter.get_text_dir()
        # For court judgments, prefer the _facts version if it exists
        facts_path = text_dir / f"{doc_id}_facts.txt"
        if facts_path.exists():
            return facts_path.read_text(encoding="utf-8")
        path = text_dir / f"{doc_id}.txt"
        if not path.exists():
            raise FileNotFoundError(
                f"Source text not found: {path}. "
                f"Ensure document '{doc_id}' exists in '{adapter.get_text_dir()}'."
            )
        return path.read_text(encoding="utf-8")

    def _list_gold_ids(self, adapter: Optional[GoldStandardAdapter] = None) -> List[str]:
        if adapter is None:
            adapter = GOLD_ADAPTERS["fir"]
        gold_dir = self._data_dir / adapter.get_gold_dir()
        if not gold_dir.exists():
            return []
        return sorted(p.stem for p in gold_dir.glob("*.json"))

    async def evaluate_single(
        self, doc_id: str, source_type: str = "fir", model_override: Optional[str] = None
    ) -> EvaluationResult:
        """Load gold JSON + text, run extraction, compare."""
        adapter = self._get_adapter(source_type)
        gold = self._load_gold(doc_id, adapter)
        text = self._load_text(doc_id, adapter)

        start = time.time()
        extracted = await self._text_extractor.extract(
            text, adapter.get_source_type(), model_override=model_override
        )
        elapsed_ms = (time.time() - start) * 1000

        entities = extracted["entities"]
        gold_items_by_type = adapter.extract_gold_items(gold)

        # Compute metrics per type
        metrics: Dict[str, EntityTypeMetrics] = {}
        for type_name, gold_list in gold_items_by_type.items():
            # Map entity types to name keys for extraction
            name_key = "description" if type_name in ("TimeEvent", "Statement") else "name"
            ext_list = _get_extracted_by_type(entities, type_name, name_key)
            m = _compute_metrics(type_name, gold_list, ext_list)
            metrics[type_name] = m

        # Aggregate across types
        total_tp = sum(m.true_positives for m in metrics.values())
        total_fp = sum(m.false_positives for m in metrics.values())
        total_fn = sum(m.false_negatives for m in metrics.values())

        agg_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        agg_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        agg_f1 = 2 * agg_p * agg_r / (agg_p + agg_r) if (agg_p + agg_r) > 0 else 0.0

        gold_count = sum(len(gl) for gl in gold_items_by_type.values())

        return EvaluationResult(
            fir_id=doc_id,
            entity_metrics=metrics,
            aggregate_precision=round(agg_p, 4),
            aggregate_recall=round(agg_r, 4),
            aggregate_f1=round(agg_f1, 4),
            extraction_time_ms=round(elapsed_ms, 1),
            gold_entity_count=gold_count,
            extracted_entity_count=len(entities),
            model_used=model_override or "gpt-4o",
        )

    async def evaluate_all(
        self, source_type: str = "fir", model_override: Optional[str] = None
    ) -> AggregateEvaluationResult:
        """Run evaluation on all gold standard files for a given source type."""
        adapter = self._get_adapter(source_type)
        doc_ids = self._list_gold_ids(adapter)
        results: List[EvaluationResult] = []

        for doc_id in doc_ids:
            try:
                result = await self.evaluate_single(doc_id, source_type, model_override)
                results.append(result)
                logger.info(
                    f"Evaluated {doc_id}: P={result.aggregate_precision:.3f} "
                    f"R={result.aggregate_recall:.3f} F1={result.aggregate_f1:.3f}"
                )
            except Exception as e:
                logger.error(f"Failed to evaluate {doc_id}: {e}")

        # Aggregate per-entity-type metrics across all cases
        per_type: Dict[str, EntityTypeMetrics] = {}
        all_types = set()
        for r in results:
            all_types.update(r.entity_metrics.keys())

        _empty = lambda t: EntityTypeMetrics(
            entity_type=t, true_positives=0, false_positives=0, false_negatives=0,
            precision=0, recall=0, f1=0
        )

        for t in sorted(all_types):
            total_tp = sum(r.entity_metrics.get(t, _empty(t)).true_positives for r in results)
            total_fp = sum(r.entity_metrics.get(t, _empty(t)).false_positives for r in results)
            total_fn = sum(r.entity_metrics.get(t, _empty(t)).false_negatives for r in results)

            p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            r_val = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * p * r_val / (p + r_val) if (p + r_val) > 0 else 0.0

            per_type[t] = EntityTypeMetrics(
                entity_type=t,
                true_positives=total_tp,
                false_positives=total_fp,
                false_negatives=total_fn,
                precision=round(p, 4),
                recall=round(r_val, 4),
                f1=round(f1, 4),
            )

        # Overall aggregation
        grand_tp = sum(m.true_positives for m in per_type.values())
        grand_fp = sum(m.false_positives for m in per_type.values())
        grand_fn = sum(m.false_negatives for m in per_type.values())

        overall_p = grand_tp / (grand_tp + grand_fp) if (grand_tp + grand_fp) > 0 else 0.0
        overall_r = grand_tp / (grand_tp + grand_fn) if (grand_tp + grand_fn) > 0 else 0.0
        overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0

        return AggregateEvaluationResult(
            total_cases=len(results),
            per_case=results,
            per_entity_type=per_type,
            overall_precision=round(overall_p, 4),
            overall_recall=round(overall_r, 4),
            overall_f1=round(overall_f1, 4),
            model_used=model_override or "gpt-4o",
        )
