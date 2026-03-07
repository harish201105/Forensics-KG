"""Generate gold standard annotations from court judgment full text using LLM."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from loguru import logger
from app.services.extraction.openai_client import OpenAIClient


# Schema for LLM-generated gold standard from court judgments
COURT_JUDGMENT_GOLD_SCHEMA = {
    "type": "object",
    "properties": {
        "case_id": {"type": "string"},
        "case_title": {"type": "string"},
        "crime_type": {"type": "string"},
        "date_of_incident": {"type": "string"},
        "date_of_judgment": {"type": "string"},
        "court": {"type": "string"},
        "judge_names": {"type": "array", "items": {"type": "string"}},
        "accused_names": {"type": "array", "items": {"type": "string"}},
        "victim_names": {"type": "array", "items": {"type": "string"}},
        "witness_names": {"type": "array", "items": {"type": "string"}},
        "lawyer_names": {"type": "array", "items": {"type": "string"}},
        "location_details": {"type": "string"},
        "locations": {"type": "array", "items": {"type": "string"}},
        "evidence_items": {"type": "array", "items": {"type": "string"}},
        "weapon_details": {"type": "string"},
        "vehicle_details": {"type": "string"},
        "legal_sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section": {"type": "string"},
                    "act": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["section", "act", "description"],
                "additionalProperties": False,
            },
        },
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
        "verdict": {"type": "string"},
        "sentence": {"type": "string"},
        "narrative_summary": {"type": "string"},
    },
    "required": [
        "case_id", "case_title", "crime_type", "judge_names",
        "accused_names", "victim_names", "witness_names", "lawyer_names",
        "locations", "evidence_items", "legal_sections",
        "timeline_events", "verdict", "sentence", "narrative_summary",
        "date_of_incident", "date_of_judgment", "court",
        "location_details", "weapon_details", "vehicle_details",
    ],
    "additionalProperties": False,
}


class GoldStandardGenerator:
    """Generate gold standard annotations from full judgment text using LLM."""

    def __init__(self, openai_client: OpenAIClient, data_dir: Path):
        self._openai = openai_client
        self._data_dir = data_dir
        self._gold_dir = data_dir / "gold_court"
        self._gold_dir.mkdir(parents=True, exist_ok=True)

    async def generate_gold(
        self, case_id: str, full_text: str, model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract comprehensive gold standard from full judgment text."""
        # Truncate very long texts to fit context window
        text_for_llm = full_text[:30000] if len(full_text) > 30000 else full_text

        system_prompt = (
            "You are a legal data extraction expert specializing in Indian criminal law. "
            "Extract ALL factual entities from the court judgment text. "
            "Be exhaustive — include every person, location, evidence item, weapon, "
            "legal section, and timeline event mentioned. "
            "For the verdict, state the outcome clearly (convicted/acquitted/remanded). "
            "For dates, use ISO format (YYYY-MM-DD) when possible."
        )
        user_prompt = (
            f"Extract comprehensive structured data from this Indian court judgment.\n"
            f"Case ID: {case_id}\n\n"
            f"Full Judgment Text:\n{text_for_llm}\n\n"
            f"Extract ALL entities including:\n"
            f"- Every person mentioned (judges, accused, victims, witnesses, lawyers)\n"
            f"- All locations mentioned in the case\n"
            f"- All evidence items discussed\n"
            f"- Weapons and vehicles if mentioned\n"
            f"- All IPC/CrPC sections cited\n"
            f"- Timeline of events\n"
            f"- The final verdict and sentence"
        )

        result = await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=COURT_JUDGMENT_GOLD_SCHEMA,
            model_override=model_override,
            temperature=0.1,
        )

        # Ensure case_id is set
        result["case_id"] = case_id

        # Save to disk
        gold_path = self._gold_dir / f"{case_id}.json"
        gold_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info(f"Generated gold standard for {case_id}: {gold_path}")

        return result

    async def generate_batch(
        self,
        judgments_dir: Optional[Path] = None,
        limit: int = 20,
        model_override: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Generate gold standards for all ingested judgments."""
        jdir = judgments_dir or self._data_dir / "court_judgments"
        results = []

        # Use full text files (not _facts files)
        files = sorted(
            p for p in jdir.glob("*.txt")
            if not p.stem.endswith("_facts")
        )

        for f in files[:limit]:
            case_id = f.stem
            # Skip if gold already exists
            gold_path = self._gold_dir / f"{case_id}.json"
            if gold_path.exists():
                logger.info(f"Gold already exists for {case_id}, skipping")
                continue

            try:
                full_text = f.read_text(encoding="utf-8")
                gold = await self.generate_gold(case_id, full_text, model_override)
                results.append(gold)
            except Exception as e:
                logger.error(f"Failed to generate gold for {case_id}: {e}")

        logger.info(f"Generated {len(results)} gold standards")
        return results

    def list_gold(self) -> List[Dict[str, str]]:
        """List all generated gold standard files."""
        results = []
        for p in sorted(self._gold_dir.glob("*.json")):
            results.append({
                "case_id": p.stem,
                "path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
        return results
