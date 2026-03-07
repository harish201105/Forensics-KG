import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from loguru import logger
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.schemas import SYNTHETIC_FIR_SCHEMA

# Pool of recurring entities so GPT-4o generates cross-case connections.
# The EntityDeduplicator merges nodes that share the same name (case-insensitive).
SHARED_ENTITY_POOL = {
    "suspects": [
        {"name": "Vikram Malhotra", "desc": "Male, 35, known gang associate, multiple prior arrests for assault and extortion"},
        {"name": "Deepak Yadav", "desc": "Male, 28, history of violent offenses, suspected in chain snatching ring"},
        {"name": "Suresh Pandey", "desc": "Male, 42, repeat offender with connections to organized crime syndicate"},
        {"name": "Priya Sharma", "desc": "Female, 31, suspected involvement in multiple fraud and forgery cases"},
        {"name": "Mohammad Irfan", "desc": "Male, 38, prior convictions for smuggling and illegal arms possession"},
    ],
    "locations": [
        {"name": "Chandni Chowk, Old Delhi", "station": "Kotwali PS, Central Delhi"},
        {"name": "Andheri West, Mumbai", "station": "DN Nagar PS, Mumbai"},
        {"name": "MG Road, Bangalore", "station": "Cubbon Park PS, Bangalore"},
        {"name": "Sector 18, Noida", "station": "Sector 20 PS, Gautam Buddh Nagar"},
        {"name": "Jubilee Hills, Hyderabad", "station": "Jubilee Hills PS, Hyderabad"},
    ],
    "weapons": [
        {"name": "Country-made pistol (.32 bore)", "type": "firearm"},
        {"name": "Butcher knife (8-inch blade)", "type": "sharp weapon"},
        {"name": "Iron rod (3 feet)", "type": "blunt weapon"},
        {"name": "Acid bottle (sulfuric acid)", "type": "chemical weapon"},
    ],
    "officers": [
        {"name": "SI Ramesh Verma", "role": "investigating officer"},
        {"name": "Inspector Kavita Singh", "role": "Crime Branch officer"},
        {"name": "ASI Bhagwan Das", "role": "forensic team lead"},
    ],
}


class SyntheticFIRGenerator:
    """Generates synthetic FIR (First Information Report) texts using GPT-4o."""

    def __init__(self, openai_client: OpenAIClient, data_dir: Optional[Path] = None):
        self._openai = openai_client
        self._data_dir = data_dir

    def _next_index(self) -> int:
        """Find the next available FIR index by scanning existing files."""
        fir_dir = self._data_dir / "fir_text" if self._data_dir else None
        if not fir_dir or not fir_dir.exists():
            return 1
        max_idx = 0
        pattern = re.compile(r"FIR-2024-(\d+)")
        for f in fir_dir.glob("FIR-2024-*.txt"):
            m = pattern.search(f.stem)
            if m:
                max_idx = max(max_idx, int(m.group(1)))
        return max_idx + 1

    @staticmethod
    def _select_shared_entities(pool: Dict[str, list]) -> str:
        """Pick random entities from the pool and return a prompt fragment."""
        parts = []
        suspects = pool.get("suspects", [])
        locations = pool.get("locations", [])
        weapons = pool.get("weapons", [])
        officers = pool.get("officers", [])

        if suspects:
            s = random.choice(suspects)
            parts.append(f"- Suspect: {s['name']} ({s.get('desc', '')})")
        if locations:
            loc = random.choice(locations)
            parts.append(
                f"- Location: {loc['name']} (Police Station: {loc.get('station', 'N/A')})"
            )
        if weapons and random.random() < 0.6:
            w = random.choice(weapons)
            parts.append(f"- Weapon: {w['name']} ({w.get('type', '')})")
        if officers and random.random() < 0.4:
            o = random.choice(officers)
            parts.append(f"- Investigating Officer: {o['name']}")

        if not parts:
            return ""
        return (
            "\nIMPORTANT — Cross-case linking: You MUST incorporate these specific "
            "entities into the FIR. Use their EXACT names as written below. "
            "Build a realistic scenario around them:\n"
            + "\n".join(parts)
            + "\n"
        )

    @staticmethod
    async def _get_existing_entities(neo4j_client: Any) -> Dict[str, list]:
        """Query the graph for existing suspects and locations to extend the pool."""
        dynamic: Dict[str, list] = {"suspects": [], "locations": []}
        try:
            persons = await neo4j_client.execute_query(
                "MATCH (p:Person) WHERE toLower(p.role) CONTAINS 'suspect' "
                "RETURN DISTINCT p.name AS name LIMIT 5"
            )
            for p in (persons or []):
                if p.get("name"):
                    dynamic["suspects"].append(
                        {"name": p["name"], "desc": "known from prior cases"}
                    )
            locs = await neo4j_client.execute_query(
                "MATCH (l:Location) RETURN DISTINCT l.name AS name LIMIT 5"
            )
            for loc in (locs or []):
                if loc.get("name"):
                    dynamic["locations"].append(
                        {"name": loc["name"], "station": "N/A"}
                    )
        except Exception as e:
            logger.warning(f"Could not fetch existing entities for pool: {e}")
        return dynamic

    @staticmethod
    def _merge_pools(base: Dict[str, list], extra: Dict[str, list]) -> Dict[str, list]:
        """Merge dynamic entities into the base pool, deduplicating by name."""
        merged: Dict[str, list] = {}
        for key in set(list(base.keys()) + list(extra.keys())):
            seen_names: set = set()
            items: list = []
            for item in base.get(key, []) + extra.get(key, []):
                name_lower = item.get("name", "").lower()
                if name_lower and name_lower not in seen_names:
                    seen_names.add(name_lower)
                    items.append(item)
            merged[key] = items
        return merged

    async def generate_batch(
        self,
        count: int = 10,
        crime_types: Optional[List[str]] = None,
        include_bloodstain: bool = True,
        neo4j_client: Any = None,
    ) -> List[Dict[str, Any]]:
        if crime_types is None:
            crime_types = [
                "homicide", "assault", "domestic_violence",
                "robbery", "hit_and_run", "kidnapping",
            ]

        # Build entity pool (static + dynamic from existing graph)
        pool = dict(SHARED_ENTITY_POOL)
        if neo4j_client:
            dynamic = await self._get_existing_entities(neo4j_client)
            pool = self._merge_pools(pool, dynamic)
            logger.info(
                f"Entity pool: {len(pool.get('suspects', []))} suspects, "
                f"{len(pool.get('locations', []))} locations"
            )

        start_index = self._next_index()
        firs = []
        for i in range(count):
            crime = crime_types[i % len(crime_types)]
            shared_prompt = self._select_shared_entities(pool)
            fir = await self._generate_single(
                start_index + i, crime, include_bloodstain, shared_prompt
            )
            fir["full_text"] = self._format_as_text(fir)
            firs.append(fir)
            logger.info(f"Generated FIR {i+1}/{count}: {fir['fir_number']}")

        return firs

    async def _generate_single(
        self,
        index: int,
        crime_type: str,
        include_bloodstain: bool,
        shared_entities_prompt: str = "",
    ) -> Dict[str, Any]:
        bloodstain_note = ""
        if include_bloodstain:
            bloodstain_note = (
                "Include detailed bloodstain pattern evidence description "
                "(pattern type from: passive_drop, transfer, projected, cast_off, "
                "impact_spatter, arterial_spurt, void, wipe, swipe, flow, pool, satellite; "
                "location; spatial distribution; estimated mechanism)."
            )

        system_prompt = (
            "You are generating realistic synthetic First Information Reports (FIR) "
            "for a forensic knowledge graph research system. Generate detailed, "
            "realistic reports with multiple named entities, specific locations, "
            "evidence items, weapons, and clear timelines. Use realistic Indian "
            "names and locations."
        )
        user_prompt = (
            f"Generate a realistic synthetic FIR report for a {crime_type} case.\n"
            f"FIR number: FIR-2024-{index:04d}\n"
            f"{bloodstain_note}\n"
            f"{shared_entities_prompt}"
            f"Requirements:\n"
            f"- At least 3 named persons (victim, suspect, witnesses)\n"
            f"- At least 2 specific locations\n"
            f"- Specific evidence items found at the scene\n"
            f"- A weapon if applicable\n"
            f"- Vehicle details if applicable\n"
            f"- Timeline with at least 4 events\n"
            f"- Detailed narrative of the incident\n"
            f"- Make it realistic and detailed."
        )
        return await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=SYNTHETIC_FIR_SCHEMA,
            temperature=0.7,
        )

    def _format_as_text(self, fir: Dict[str, Any]) -> str:
        witnesses = "\n".join(
            f"  - {s}" for s in fir.get("witness_statements", [])
        )
        timeline = "\n".join(
            f"  - {e['time']}: {e['description']}"
            for e in fir.get("timeline_events", [])
        )
        return (
            f"FIRST INFORMATION REPORT\n"
            f"{'='*50}\n"
            f"FIR No: {fir['fir_number']}\n"
            f"Date: {fir['date_filed']}\n"
            f"Police Station: {fir.get('police_station', 'N/A')}\n"
            f"Crime Type: {fir['crime_type']}\n\n"
            f"COMPLAINANT: {fir['complainant_name']}\n\n"
            f"VICTIM: {fir.get('victim_name', 'N/A')}\n"
            f"  {fir.get('victim_description', '')}\n\n"
            f"NARRATIVE:\n{fir['narrative']}\n\n"
            f"LOCATION: {fir['location_details']}\n\n"
            f"SUSPECT DESCRIPTION:\n{fir.get('suspect_description', 'Unknown')}\n\n"
            f"EVIDENCE:\n{fir.get('evidence_description', 'None recorded')}\n\n"
            f"BLOODSTAIN EVIDENCE:\n{fir.get('bloodstain_evidence', 'None')}\n\n"
            f"WEAPON: {fir.get('weapon_details', 'None')}\n\n"
            f"VEHICLE: {fir.get('vehicle_details', 'None')}\n\n"
            f"WITNESS STATEMENTS:\n{witnesses}\n\n"
            f"TIMELINE OF EVENTS:\n{timeline}\n"
        )

    async def save_batch(self, firs: List[Dict], output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        gold_dir = output_dir.parent / "gold"
        gold_dir.mkdir(parents=True, exist_ok=True)

        for fir in firs:
            fir_num = fir["fir_number"]
            # Save text version
            text_path = output_dir / f"{fir_num}.txt"
            text_path.write_text(fir["full_text"], encoding="utf-8")
            # Save gold JSON (structured data = ground truth)
            gold_path = gold_dir / f"{fir_num}.json"
            gold_data = {k: v for k, v in fir.items() if k != "full_text"}
            gold_path.write_text(
                json.dumps(gold_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        logger.info(
            f"Saved {len(firs)} FIRs to {output_dir} and gold to {gold_dir}"
        )
