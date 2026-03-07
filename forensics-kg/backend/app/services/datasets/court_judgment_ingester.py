"""Ingest real criminal case judgments from Indian court data sources."""

import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


# HuggingFace dataset REST API (no auth required)
_HF_DATASET = "rishiai/indian-court-judgements-and-its-summaries"
_HF_API_URL = (
    "https://datasets-server.huggingface.co/rows"
    f"?dataset={_HF_DATASET}&config=default&split=train"
)


class CourtJudgmentIngester:
    """Download and process Indian criminal court judgments.

    Supports three data sources:
    1. HuggingFace dataset (6,940 judgments, auto-downloaded, no auth needed)
    2. Indian Supreme Court Judgments dataset (Kaggle CSV, manual download)
    3. Indian Kanoon API (api.indiankanoon.org, requires auth token)
    """

    # Criminal-related keywords for filtering
    CRIMINAL_KEYWORDS = [
        "murder", "homicide", "assault", "robbery", "theft",
        "kidnapping", "rape", "dowry", "fir", "ipc", "crpc",
        "accused", "convicted", "acquitted", "culpable homicide",
        "criminal conspiracy", "Section 302", "Section 307",
        "Section 376", "Section 420", "Section 498",
    ]

    def __init__(self, data_dir: Path, indian_kanoon_token: str = ""):
        self._data_dir = data_dir
        self._judgments_dir = data_dir / "court_judgments"
        self._judgments_dir.mkdir(parents=True, exist_ok=True)
        self._ik_token = indian_kanoon_token

    # ── HuggingFace (recommended, no auth needed) ─────────────────────

    async def ingest_from_huggingface(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Download and ingest criminal judgments from HuggingFace dataset.

        Uses the HuggingFace datasets-server REST API to fetch rows directly.
        No authentication or extra libraries required — only httpx.
        Dataset: rishiai/indian-court-judgements-and-its-summaries (6,940 judgments)
        """
        if httpx is None:
            raise ImportError(
                "httpx is required for HuggingFace download. Install with: pip install httpx"
            )

        judgments: List[Dict[str, Any]] = []
        offset = 0
        batch_size = 100  # HF API max per request
        total_scanned = 0

        logger.info(f"Downloading criminal judgments from HuggingFace ({_HF_DATASET})...")

        async with httpx.AsyncClient(timeout=60.0) as client:
            while len(judgments) < limit:
                url = f"{_HF_API_URL}&offset={offset}&length={batch_size}"
                try:
                    resp = await client.get(url)
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.error(f"HuggingFace API error at offset {offset}: {e}")
                    break

                rows = data.get("rows", [])
                if not rows:
                    logger.info(f"No more rows at offset {offset}, ending scan")
                    break

                for row_obj in rows:
                    row = row_obj.get("row", {})
                    total_scanned += 1

                    text = row.get("Judgment", row.get("judgment", ""))
                    if not text or len(text) < 500:
                        continue

                    # Filter for criminal cases
                    if not self._is_criminal_case(text, ""):
                        continue

                    facts = self._extract_facts_section(text)
                    if not facts or len(facts) < 200:
                        continue

                    row_id = row.get("ID", row.get("id", total_scanned))
                    judgment = {
                        "case_id": f"HF-{row_id:04d}" if isinstance(row_id, int) else f"HF-{row_id}",
                        "title": text[:120].replace("\n", " ").strip(),
                        "full_text": text,
                        "facts_section": facts,
                        "date": "",
                        "source": "huggingface",
                    }
                    judgments.append(judgment)

                    if len(judgments) >= limit:
                        break

                offset += batch_size
                logger.info(
                    f"Scanned {total_scanned} rows, found {len(judgments)} criminal judgments so far"
                )

        # Save to disk
        for j in judgments:
            self._save_judgment(j)

        logger.info(
            f"Ingested {len(judgments)} criminal judgments from HuggingFace "
            f"(scanned {total_scanned} total)"
        )
        return judgments

    # ── CSV (Kaggle manual download) ──────────────────────────────────

    async def ingest_from_csv(
        self, csv_path: Path, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Ingest judgments from a downloaded Kaggle CSV file.

        Expected CSV columns: case_id/id, case_title/title, judgment_text/text, date, ...
        The CSV should be downloaded manually from Kaggle and placed in data/.
        """
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV file not found at {csv_path}. "
                f"Download the Indian Supreme Court Judgments dataset from Kaggle "
                f"and place the CSV in {self._data_dir}/"
            )

        judgments = []
        with open(csv_path, encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if len(judgments) >= limit:
                    break

                # Normalize column names (different datasets use different names)
                text = (
                    row.get("judgment_text")
                    or row.get("text")
                    or row.get("judgment")
                    or row.get("Judgment")
                    or row.get("body")
                    or ""
                )
                title = (
                    row.get("case_title")
                    or row.get("title")
                    or row.get("case_name")
                    or ""
                )
                case_id = (
                    row.get("case_id")
                    or row.get("id")
                    or row.get("ID")
                    or row.get("serial_no")
                    or f"SC-{len(judgments)+1:04d}"
                )

                if not text or len(text) < 500:
                    continue

                # Filter for criminal cases
                if not self._is_criminal_case(text, title):
                    continue

                facts = self._extract_facts_section(text)
                if not facts or len(facts) < 200:
                    continue

                judgment = {
                    "case_id": str(case_id).strip(),
                    "title": title.strip(),
                    "full_text": text,
                    "facts_section": facts,
                    "date": row.get("date", row.get("judgment_date", "")),
                    "source": "kaggle_csv",
                }
                judgments.append(judgment)

        # Save to disk
        for j in judgments:
            self._save_judgment(j)

        logger.info(f"Ingested {len(judgments)} criminal judgments from CSV")
        return judgments

    # ── Indian Kanoon API ─────────────────────────────────────────────

    async def ingest_from_indian_kanoon(
        self, query: str = "murder FIR IPC", limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Search and download judgments from Indian Kanoon API."""
        if httpx is None:
            raise ImportError("httpx is required for Indian Kanoon API. Install with: pip install httpx")

        if not self._ik_token:
            raise ValueError(
                "Indian Kanoon API requires an auth token. "
                "Set INDIAN_KANOON_API_TOKEN in your .env file. "
                "Get a token from https://api.indiankanoon.org/"
            )

        judgments = []
        page = 0

        async with httpx.AsyncClient(timeout=30.0) as client:
            while len(judgments) < limit:
                # Search endpoint
                search_url = "https://api.indiankanoon.org/search/"
                params = {"formInput": query, "pagenum": page}
                headers = {"Authorization": f"Token {self._ik_token}"}

                try:
                    resp = await client.post(search_url, data=params, headers=headers)
                    if resp.status_code == 403:
                        logger.warning(
                            "Indian Kanoon API returned 403. Check your INDIAN_KANOON_API_TOKEN."
                        )
                        break
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    logger.error(f"Indian Kanoon API error: {e}")
                    break

                docs = data.get("docs", [])
                if not docs:
                    break

                for doc in docs:
                    if len(judgments) >= limit:
                        break

                    doc_id = doc.get("tid", "")
                    title = doc.get("title", "")

                    # Fetch full document
                    try:
                        doc_url = f"https://api.indiankanoon.org/doc/{doc_id}/"
                        doc_resp = await client.post(doc_url, headers=headers)
                        doc_resp.raise_for_status()
                        doc_data = doc_resp.json()
                        full_text = doc_data.get("doc", "")
                    except Exception as e:
                        logger.warning(f"Failed to fetch doc {doc_id}: {e}")
                        continue

                    if not full_text or len(full_text) < 500:
                        continue

                    # Strip HTML tags
                    full_text = re.sub(r"<[^>]+>", " ", full_text)
                    full_text = re.sub(r"\s+", " ", full_text).strip()

                    facts = self._extract_facts_section(full_text)
                    if not facts or len(facts) < 200:
                        continue

                    judgment = {
                        "case_id": f"IK-{doc_id}",
                        "title": title,
                        "full_text": full_text,
                        "facts_section": facts,
                        "date": doc.get("publishdate", ""),
                        "source": "indian_kanoon",
                    }
                    judgments.append(judgment)

                page += 1

        for j in judgments:
            self._save_judgment(j)

        logger.info(f"Ingested {len(judgments)} judgments from Indian Kanoon")
        return judgments

    # ── Listing ───────────────────────────────────────────────────────

    def list_ingested(self) -> List[Dict[str, str]]:
        """List all ingested judgment files (excludes _facts.txt files)."""
        results = []
        for p in sorted(self._judgments_dir.glob("*.txt")):
            if p.stem.endswith("_facts"):
                continue
            results.append({
                "case_id": p.stem,
                "path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
        return results

    # ── Internal helpers ──────────────────────────────────────────────

    def _is_criminal_case(self, text: str, title: str) -> bool:
        """Check if a judgment is criminal in nature."""
        combined = (text[:3000] + " " + title).lower()
        matches = sum(1 for kw in self.CRIMINAL_KEYWORDS if kw.lower() in combined)
        return matches >= 2

    def _extract_facts_section(self, text: str) -> str:
        """Extract the 'Facts of the Case' section from judgment text."""
        # Common heading patterns in Indian judgments
        patterns = [
            r"(?:FACTS?\s+(?:OF\s+THE\s+)?CASE|FACTUAL\s+BACKGROUND|BRIEF\s+FACTS|"
            r"PROSECUTION\s+(?:CASE|STORY)|CASE\s+OF\s+THE\s+PROSECUTION)\s*[:\-.]?\s*\n([\s\S]+?)(?="
            r"\n\s*(?:ARGUMENTS?|SUBMISSIONS?|CONTENTIONS?|ISSUES?|QUESTIONS?\s+(?:OF|FOR)|"
            r"DISCUSSION|ANALYSIS|REASONING|FINDINGS?|CONCLUSION|JUDGMENT|ORDER)\s*[:\-.]?\s*\n)",

            r"(?:\d+\.\s+)?(?:The\s+)?(?:brief\s+)?facts\s+(?:of\s+the\s+case\s+)?(?:are|in\s+brief)[:\s]+"
            r"([\s\S]{200,3000}?)(?:\n\s*\d+\.|\n\s*(?:It\s+is|The\s+learned|We\s+have))",
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                facts = match.group(1).strip()
                if len(facts) > 200:
                    return facts

        # Fallback: take the first substantial paragraphs after initial metadata
        lines = text.split("\n")
        start_idx = 0
        for i, line in enumerate(lines):
            # Skip initial case citation / header lines
            if len(line.strip()) > 100 and not line.strip().startswith("("):
                start_idx = i
                break

        # Take ~3000 chars from start
        result = "\n".join(lines[start_idx:])
        return result[:3000].strip() if len(result) > 200 else ""

    def _save_judgment(self, judgment: Dict[str, Any]) -> None:
        """Save judgment text and facts to disk."""
        case_id = judgment["case_id"]
        # Save full text
        full_path = self._judgments_dir / f"{case_id}.txt"
        full_path.write_text(judgment["full_text"], encoding="utf-8")
        # Save facts section separately
        facts_path = self._judgments_dir / f"{case_id}_facts.txt"
        facts_path.write_text(judgment["facts_section"], encoding="utf-8")
