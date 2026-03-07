"""Synthetic document generators for forensic document types."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from app.services.extraction.openai_client import OpenAIClient


# --- Synthetic Document Schemas ---

SYNTHETIC_POSTMORTEM_SCHEMA = {
    "type": "object",
    "properties": {
        "report_id": {"type": "string"},
        "date_of_examination": {"type": "string"},
        "deceased_name": {"type": "string"},
        "deceased_age": {"type": "string"},
        "deceased_gender": {"type": "string"},
        "examining_doctor": {"type": "string"},
        "hospital": {"type": "string"},
        "case_reference": {"type": "string"},
        "external_examination": {"type": "string"},
        "injuries": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "injury_number": {"type": "integer"},
                    "type": {"type": "string"},
                    "location": {"type": "string"},
                    "dimensions": {"type": "string"},
                    "description": {"type": "string"},
                    "estimated_age": {"type": "string"},
                },
                "required": ["injury_number", "type", "location", "dimensions", "description", "estimated_age"],
                "additionalProperties": False,
            },
        },
        "internal_examination": {"type": "string"},
        "organ_findings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "organ": {"type": "string"},
                    "condition": {"type": "string"},
                    "weight": {"type": "string"},
                },
                "required": ["organ", "condition", "weight"],
                "additionalProperties": False,
            },
        },
        "toxicology_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "substance": {"type": "string"},
                    "result": {"type": "string"},
                    "concentration": {"type": "string"},
                },
                "required": ["substance", "result", "concentration"],
                "additionalProperties": False,
            },
        },
        "cause_of_death": {"type": "string"},
        "manner_of_death": {"type": "string"},
        "estimated_time_of_death": {"type": "string"},
        "narrative": {"type": "string"},
    },
    "required": [
        "report_id", "date_of_examination", "deceased_name", "deceased_age",
        "deceased_gender", "examining_doctor", "hospital", "case_reference",
        "external_examination", "injuries", "internal_examination",
        "organ_findings", "toxicology_results", "cause_of_death",
        "manner_of_death", "estimated_time_of_death", "narrative",
    ],
    "additionalProperties": False,
}

SYNTHETIC_LAB_REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "report_id": {"type": "string"},
        "lab_name": {"type": "string"},
        "analyst_name": {"type": "string"},
        "date_received": {"type": "string"},
        "date_reported": {"type": "string"},
        "case_reference": {"type": "string"},
        "requesting_officer": {"type": "string"},
        "samples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "sample_type": {"type": "string"},
                    "source": {"type": "string"},
                    "collection_date": {"type": "string"},
                    "condition": {"type": "string"},
                },
                "required": ["sample_id", "sample_type", "source", "collection_date", "condition"],
                "additionalProperties": False,
            },
        },
        "tests_performed": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "test_name": {"type": "string"},
                    "sample_id": {"type": "string"},
                    "method": {"type": "string"},
                    "result": {"type": "string"},
                    "interpretation": {"type": "string"},
                },
                "required": ["test_name", "sample_id", "method", "result", "interpretation"],
                "additionalProperties": False,
            },
        },
        "dna_profiles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "sample_id": {"type": "string"},
                    "profile_type": {"type": "string"},
                    "loci_tested": {"type": "string"},
                    "alleles": {"type": "string"},
                    "match_result": {"type": "string"},
                    "match_probability": {"type": "string"},
                },
                "required": ["sample_id", "profile_type", "loci_tested", "alleles", "match_result", "match_probability"],
                "additionalProperties": False,
            },
        },
        "conclusions": {"type": "string"},
        "narrative": {"type": "string"},
    },
    "required": [
        "report_id", "lab_name", "analyst_name", "date_received",
        "date_reported", "case_reference", "requesting_officer",
        "samples", "tests_performed", "dna_profiles", "conclusions", "narrative",
    ],
    "additionalProperties": False,
}

SYNTHETIC_DEPOSITION_SCHEMA = {
    "type": "object",
    "properties": {
        "deposition_id": {"type": "string"},
        "case_reference": {"type": "string"},
        "witness_name": {"type": "string"},
        "witness_age": {"type": "string"},
        "witness_occupation": {"type": "string"},
        "witness_relation": {"type": "string"},
        "recording_officer": {"type": "string"},
        "date_recorded": {"type": "string"},
        "location_recorded": {"type": "string"},
        "statement_text": {"type": "string"},
        "key_claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "context": {"type": "string"},
                    "type": {"type": "string"},
                },
                "required": ["claim", "context", "type"],
                "additionalProperties": False,
            },
        },
        "persons_mentioned": {"type": "array", "items": {"type": "string"}},
        "locations_mentioned": {"type": "array", "items": {"type": "string"}},
        "timeline_events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "time": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["time", "description"],
                "additionalProperties": False,
            },
        },
        "narrative": {"type": "string"},
    },
    "required": [
        "deposition_id", "case_reference", "witness_name", "witness_age",
        "witness_occupation", "witness_relation", "recording_officer",
        "date_recorded", "location_recorded", "statement_text",
        "key_claims", "persons_mentioned", "locations_mentioned",
        "timeline_events", "narrative",
    ],
    "additionalProperties": False,
}

SYNTHETIC_SOCO_SCHEMA = {
    "type": "object",
    "properties": {
        "report_id": {"type": "string"},
        "case_reference": {"type": "string"},
        "scene_officer": {"type": "string"},
        "date_of_visit": {"type": "string"},
        "arrival_time": {"type": "string"},
        "departure_time": {"type": "string"},
        "scene_type": {"type": "string"},
        "scene_address": {"type": "string"},
        "scene_description": {"type": "string"},
        "weather_conditions": {"type": "string"},
        "evidence_collected": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "item_number": {"type": "integer"},
                    "description": {"type": "string"},
                    "location_in_scene": {"type": "string"},
                    "condition": {"type": "string"},
                    "collection_method": {"type": "string"},
                    "packaging": {"type": "string"},
                },
                "required": ["item_number", "description", "location_in_scene", "condition", "collection_method", "packaging"],
                "additionalProperties": False,
            },
        },
        "chain_of_custody": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "handler": {"type": "string"},
                    "action": {"type": "string"},
                    "timestamp": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["handler", "action", "timestamp", "location"],
                "additionalProperties": False,
            },
        },
        "photographs_taken": {"type": "integer"},
        "sketches_made": {"type": "integer"},
        "observations": {"type": "string"},
        "narrative": {"type": "string"},
    },
    "required": [
        "report_id", "case_reference", "scene_officer", "date_of_visit",
        "arrival_time", "departure_time", "scene_type", "scene_address",
        "scene_description", "weather_conditions", "evidence_collected",
        "chain_of_custody", "photographs_taken", "sketches_made",
        "observations", "narrative",
    ],
    "additionalProperties": False,
}


class PostMortemGenerator:
    """Generate synthetic post-mortem / autopsy examination reports."""

    def __init__(self, openai_client: OpenAIClient):
        self._openai = openai_client

    async def generate_batch(
        self, count: int = 10, crime_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        if crime_types is None:
            crime_types = ["homicide", "assault", "poisoning", "hit_and_run", "stabbing", "gunshot"]

        reports = []
        for i in range(count):
            crime = crime_types[i % len(crime_types)]
            report = await self._generate_single(i + 1, crime)
            report["full_text"] = self._format_as_text(report)
            reports.append(report)
            logger.info(f"Generated post-mortem {i+1}/{count}: {report['report_id']}")
        return reports

    async def _generate_single(self, index: int, crime_type: str) -> Dict[str, Any]:
        system_prompt = (
            "You are generating realistic synthetic post-mortem examination reports "
            "for a forensic knowledge graph research system. Create detailed, medically "
            "accurate reports with specific injury descriptions, organ findings, and "
            "toxicology results. Use realistic Indian names and hospital details."
        )
        user_prompt = (
            f"Generate a realistic synthetic post-mortem report for a {crime_type} case.\n"
            f"Report ID: PM-2024-{index:04d}\n"
            f"Requirements:\n"
            f"- At least 3 distinct injuries with precise anatomical locations and dimensions\n"
            f"- At least 3 organ findings\n"
            f"- Toxicology results (at least 2 substances tested)\n"
            f"- Estimated time of death\n"
            f"- Clear cause and manner of death\n"
            f"- Reference to a related FIR case"
        )
        return await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=SYNTHETIC_POSTMORTEM_SCHEMA,
            temperature=0.7,
        )

    def _format_as_text(self, report: Dict[str, Any]) -> str:
        injuries_text = "\n".join(
            f"  Injury #{inj['injury_number']}: {inj['type']} at {inj['location']}, "
            f"{inj['dimensions']}, {inj['description']} (Age: {inj['estimated_age']})"
            for inj in report.get("injuries", [])
        )
        organs_text = "\n".join(
            f"  {org['organ']}: {org['condition']} (Weight: {org['weight']})"
            for org in report.get("organ_findings", [])
        )
        tox_text = "\n".join(
            f"  {t['substance']}: {t['result']} ({t['concentration']})"
            for t in report.get("toxicology_results", [])
        )
        return (
            f"POST-MORTEM EXAMINATION REPORT\n"
            f"{'='*50}\n"
            f"Report No: {report['report_id']}\n"
            f"Date: {report['date_of_examination']}\n"
            f"Hospital: {report.get('hospital', 'N/A')}\n"
            f"Case Ref: {report.get('case_reference', 'N/A')}\n\n"
            f"DECEASED: {report['deceased_name']}, Age: {report['deceased_age']}, "
            f"Gender: {report['deceased_gender']}\n\n"
            f"EXAMINING DOCTOR: {report['examining_doctor']}\n\n"
            f"EXTERNAL EXAMINATION:\n{report.get('external_examination', '')}\n\n"
            f"INJURIES:\n{injuries_text}\n\n"
            f"INTERNAL EXAMINATION:\n{report.get('internal_examination', '')}\n\n"
            f"ORGAN FINDINGS:\n{organs_text}\n\n"
            f"TOXICOLOGY:\n{tox_text}\n\n"
            f"CAUSE OF DEATH: {report.get('cause_of_death', 'Undetermined')}\n"
            f"MANNER OF DEATH: {report.get('manner_of_death', 'Undetermined')}\n"
            f"ESTIMATED TIME OF DEATH: {report.get('estimated_time_of_death', 'N/A')}\n"
        )

    async def save_batch(self, reports: List[Dict], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        gold_dir = output_dir.parent / "gold_postmortem"
        gold_dir.mkdir(parents=True, exist_ok=True)
        for r in reports:
            rid = r["report_id"]
            (output_dir / f"{rid}.txt").write_text(r["full_text"], encoding="utf-8")
            gold_data = {k: v for k, v in r.items() if k != "full_text"}
            (gold_dir / f"{rid}.json").write_text(
                json.dumps(gold_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        logger.info(f"Saved {len(reports)} post-mortem reports")


class LabReportGenerator:
    """Generate synthetic forensic laboratory reports."""

    def __init__(self, openai_client: OpenAIClient):
        self._openai = openai_client

    async def generate_batch(self, count: int = 10) -> List[Dict[str, Any]]:
        report_types = ["blood_analysis", "dna_profiling", "drug_testing", "fiber_analysis", "ballistics_residue", "fingerprint_chemical"]
        reports = []
        for i in range(count):
            rtype = report_types[i % len(report_types)]
            report = await self._generate_single(i + 1, rtype)
            report["full_text"] = self._format_as_text(report)
            reports.append(report)
            logger.info(f"Generated lab report {i+1}/{count}: {report['report_id']}")
        return reports

    async def _generate_single(self, index: int, report_type: str) -> Dict[str, Any]:
        system_prompt = (
            "You are generating realistic synthetic forensic laboratory analysis reports. "
            "Include specific test methods, sample descriptions, quantitative results, "
            "and DNA profiles where applicable. Use realistic Indian names and lab details."
        )
        user_prompt = (
            f"Generate a realistic forensic lab report focusing on {report_type}.\n"
            f"Report ID: LAB-2024-{index:04d}\n"
            f"Requirements:\n"
            f"- At least 3 samples with detailed descriptions\n"
            f"- At least 3 tests performed with methods and results\n"
            f"- DNA profile data if applicable\n"
            f"- Clear conclusions linking findings to the case"
        )
        return await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=SYNTHETIC_LAB_REPORT_SCHEMA,
            temperature=0.7,
        )

    def _format_as_text(self, report: Dict[str, Any]) -> str:
        samples_text = "\n".join(
            f"  {s['sample_id']}: {s['sample_type']} from {s['source']} "
            f"(Collected: {s['collection_date']}, Condition: {s['condition']})"
            for s in report.get("samples", [])
        )
        tests_text = "\n".join(
            f"  {t['test_name']} on {t['sample_id']}: {t['result']} "
            f"(Method: {t['method']}, Interpretation: {t['interpretation']})"
            for t in report.get("tests_performed", [])
        )
        dna_text = "\n".join(
            f"  {d['sample_id']}: {d['profile_type']}, Loci: {d['loci_tested']}, "
            f"Match: {d['match_result']} (P={d['match_probability']})"
            for d in report.get("dna_profiles", [])
        )
        return (
            f"FORENSIC LABORATORY REPORT\n"
            f"{'='*50}\n"
            f"Report No: {report['report_id']}\n"
            f"Lab: {report.get('lab_name', 'N/A')}\n"
            f"Analyst: {report.get('analyst_name', 'N/A')}\n"
            f"Date Received: {report.get('date_received', 'N/A')}\n"
            f"Date Reported: {report.get('date_reported', 'N/A')}\n"
            f"Case Ref: {report.get('case_reference', 'N/A')}\n\n"
            f"SAMPLES:\n{samples_text}\n\n"
            f"TESTS PERFORMED:\n{tests_text}\n\n"
            f"DNA PROFILES:\n{dna_text}\n\n"
            f"CONCLUSIONS:\n{report.get('conclusions', '')}\n"
        )

    async def save_batch(self, reports: List[Dict], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        gold_dir = output_dir.parent / "gold_lab"
        gold_dir.mkdir(parents=True, exist_ok=True)
        for r in reports:
            rid = r["report_id"]
            (output_dir / f"{rid}.txt").write_text(r["full_text"], encoding="utf-8")
            gold_data = {k: v for k, v in r.items() if k != "full_text"}
            (gold_dir / f"{rid}.json").write_text(
                json.dumps(gold_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        logger.info(f"Saved {len(reports)} lab reports")


class WitnessDepositionGenerator:
    """Generate synthetic witness deposition / statement records."""

    def __init__(self, openai_client: OpenAIClient):
        self._openai = openai_client

    async def generate_batch(self, count: int = 10) -> List[Dict[str, Any]]:
        witness_types = ["eyewitness", "character_witness", "alibi_witness", "expert_witness", "hostile_witness", "complainant"]
        reports = []
        for i in range(count):
            wtype = witness_types[i % len(witness_types)]
            report = await self._generate_single(i + 1, wtype)
            report["full_text"] = self._format_as_text(report)
            reports.append(report)
            logger.info(f"Generated deposition {i+1}/{count}: {report['deposition_id']}")
        return reports

    async def _generate_single(self, index: int, witness_type: str) -> Dict[str, Any]:
        system_prompt = (
            "You are generating realistic synthetic witness deposition records "
            "for Indian criminal cases. Include specific details the witness observed, "
            "timeline events, and persons mentioned. Use realistic Indian names."
        )
        user_prompt = (
            f"Generate a realistic witness deposition for a {witness_type}.\n"
            f"Deposition ID: DEP-2024-{index:04d}\n"
            f"Requirements:\n"
            f"- Detailed statement describing what the witness saw/knew\n"
            f"- At least 3 key claims with context\n"
            f"- At least 3 persons mentioned by name\n"
            f"- At least 3 timeline events\n"
            f"- If alibi witness, include alibi details\n"
            f"- Include potential contradictions or uncertainties"
        )
        return await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=SYNTHETIC_DEPOSITION_SCHEMA,
            temperature=0.7,
        )

    def _format_as_text(self, report: Dict[str, Any]) -> str:
        claims_text = "\n".join(
            f"  - [{c['type']}] {c['claim']} (Context: {c['context']})"
            for c in report.get("key_claims", [])
        )
        timeline_text = "\n".join(
            f"  - {e['time']}: {e['description']}"
            for e in report.get("timeline_events", [])
        )
        return (
            f"WITNESS DEPOSITION RECORD\n"
            f"{'='*50}\n"
            f"Deposition ID: {report['deposition_id']}\n"
            f"Case Ref: {report.get('case_reference', 'N/A')}\n"
            f"Date: {report.get('date_recorded', 'N/A')}\n"
            f"Location: {report.get('location_recorded', 'N/A')}\n\n"
            f"WITNESS: {report['witness_name']}, Age: {report.get('witness_age', 'N/A')}\n"
            f"Occupation: {report.get('witness_occupation', 'N/A')}\n"
            f"Relation to Case: {report.get('witness_relation', 'N/A')}\n\n"
            f"RECORDING OFFICER: {report.get('recording_officer', 'N/A')}\n\n"
            f"STATEMENT:\n{report.get('statement_text', '')}\n\n"
            f"KEY CLAIMS:\n{claims_text}\n\n"
            f"TIMELINE:\n{timeline_text}\n"
        )

    async def save_batch(self, reports: List[Dict], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        gold_dir = output_dir.parent / "gold_depositions"
        gold_dir.mkdir(parents=True, exist_ok=True)
        for r in reports:
            rid = r["deposition_id"]
            (output_dir / f"{rid}.txt").write_text(r["full_text"], encoding="utf-8")
            gold_data = {k: v for k, v in r.items() if k != "full_text"}
            (gold_dir / f"{rid}.json").write_text(
                json.dumps(gold_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        logger.info(f"Saved {len(reports)} depositions")


class SOCOReportGenerator:
    """Generate synthetic Scene of Crime Officer (SOCO) reports."""

    def __init__(self, openai_client: OpenAIClient):
        self._openai = openai_client

    async def generate_batch(self, count: int = 10) -> List[Dict[str, Any]]:
        scene_types = ["indoor_residence", "outdoor_street", "vehicle", "commercial_premises", "open_field", "construction_site"]
        reports = []
        for i in range(count):
            stype = scene_types[i % len(scene_types)]
            report = await self._generate_single(i + 1, stype)
            report["full_text"] = self._format_as_text(report)
            reports.append(report)
            logger.info(f"Generated SOCO report {i+1}/{count}: {report['report_id']}")
        return reports

    async def _generate_single(self, index: int, scene_type: str) -> Dict[str, Any]:
        system_prompt = (
            "You are generating realistic synthetic Scene of Crime Officer (SOCO) reports "
            "for Indian criminal cases. Include detailed scene descriptions, evidence "
            "collection procedures, chain of custody, and observations. Use realistic "
            "Indian names and addresses."
        )
        user_prompt = (
            f"Generate a realistic SOCO report for a {scene_type} crime scene.\n"
            f"Report ID: SOCO-2024-{index:04d}\n"
            f"Requirements:\n"
            f"- Detailed scene description with dimensions and conditions\n"
            f"- At least 5 evidence items with specific locations and collection methods\n"
            f"- Chain of custody entries (at least 3)\n"
            f"- Weather conditions at time of visit\n"
            f"- Number of photographs and sketches\n"
            f"- Detailed observations about the scene"
        )
        return await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=SYNTHETIC_SOCO_SCHEMA,
            temperature=0.7,
        )

    def _format_as_text(self, report: Dict[str, Any]) -> str:
        evidence_text = "\n".join(
            f"  Item #{e['item_number']}: {e['description']} at {e['location_in_scene']}\n"
            f"    Condition: {e['condition']}, Method: {e['collection_method']}, "
            f"Packaging: {e['packaging']}"
            for e in report.get("evidence_collected", [])
        )
        custody_text = "\n".join(
            f"  {c['timestamp']}: {c['handler']} - {c['action']} at {c['location']}"
            for c in report.get("chain_of_custody", [])
        )
        return (
            f"SCENE OF CRIME OFFICER REPORT\n"
            f"{'='*50}\n"
            f"Report No: {report['report_id']}\n"
            f"Case Ref: {report.get('case_reference', 'N/A')}\n"
            f"Scene Officer: {report.get('scene_officer', 'N/A')}\n"
            f"Date: {report.get('date_of_visit', 'N/A')}\n"
            f"Arrival: {report.get('arrival_time', 'N/A')}, "
            f"Departure: {report.get('departure_time', 'N/A')}\n\n"
            f"SCENE: {report.get('scene_type', 'N/A')}\n"
            f"Address: {report.get('scene_address', 'N/A')}\n"
            f"Weather: {report.get('weather_conditions', 'N/A')}\n\n"
            f"SCENE DESCRIPTION:\n{report.get('scene_description', '')}\n\n"
            f"EVIDENCE COLLECTED:\n{evidence_text}\n\n"
            f"CHAIN OF CUSTODY:\n{custody_text}\n\n"
            f"DOCUMENTATION: {report.get('photographs_taken', 0)} photographs, "
            f"{report.get('sketches_made', 0)} sketches\n\n"
            f"OBSERVATIONS:\n{report.get('observations', '')}\n"
        )

    async def save_batch(self, reports: List[Dict], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        gold_dir = output_dir.parent / "gold_soco"
        gold_dir.mkdir(parents=True, exist_ok=True)
        for r in reports:
            rid = r["report_id"]
            (output_dir / f"{rid}.txt").write_text(r["full_text"], encoding="utf-8")
            gold_data = {k: v for k, v in r.items() if k != "full_text"}
            (gold_dir / f"{rid}.json").write_text(
                json.dumps(gold_data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        logger.info(f"Saved {len(reports)} SOCO reports")
