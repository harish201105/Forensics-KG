"""Timeline analysis: reconstruct event timelines from TimeEvent nodes."""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from loguru import logger
from dateutil import parser as dateutil_parser

from app.services.graph.neo4j_client import Neo4jClient


class TemporalAnalyzer:
    """Reconstruct and analyze timelines from TimeEvent nodes in the KG."""

    def __init__(self, neo4j_client: Neo4jClient):
        self._neo4j = neo4j_client

    def _parse_timestamp(self, ts: str) -> Optional[datetime]:
        """Parse timestamp string, handling multiple formats."""
        if not ts:
            return None
        try:
            return dateutil_parser.parse(ts)
        except (ValueError, TypeError):
            logger.warning(f"Could not parse timestamp: {ts}")
            return None

    async def get_case_timeline(self, case_id: str) -> Dict[str, Any]:
        """Query TimeEvent nodes for a case, sort chronologically."""
        query = """
        MATCH (c:Case {case_id: $case_id})-[:HAPPENED_ON]->(te:TimeEvent)
        RETURN properties(te) as event
        """
        results = await self._neo4j.execute_query(query, {"case_id": case_id})

        if not results:
            # Try matching via relationships from other connected entities
            query2 = """
            MATCH (c:Case {case_id: $case_id})-[*1..2]-(te:TimeEvent)
            RETURN DISTINCT properties(te) as event
            """
            results = await self._neo4j.execute_query(query2, {"case_id": case_id})

        events = []
        for r in results:
            event = r.get("event", {})
            ts_str = event.get("timestamp") or event.get("date") or event.get("time") or ""
            parsed = self._parse_timestamp(ts_str)
            events.append({
                "event_id": event.get("event_id", ""),
                "timestamp": ts_str,
                "parsed_timestamp": parsed.isoformat() if parsed else None,
                "description": event.get("description", ""),
                "precision": event.get("precision", ""),
            })

        # Sort by parsed timestamp
        events.sort(key=lambda e: e["parsed_timestamp"] or "9999")

        return {
            "case_id": case_id,
            "events": events,
            "total_events": len(events),
        }

    async def detect_inconsistencies(self, case_id: str) -> List[Dict[str, Any]]:
        """Find temporal gaps and inconsistencies in the timeline."""
        timeline = await self.get_case_timeline(case_id)
        events = timeline["events"]
        inconsistencies: List[Dict[str, Any]] = []

        # Check for gaps > 2 hours between consecutive events
        for i in range(len(events) - 1):
            ts1 = self._parse_timestamp(events[i]["timestamp"])
            ts2 = self._parse_timestamp(events[i + 1]["timestamp"])
            if ts1 and ts2:
                gap = ts2 - ts1
                if gap > timedelta(hours=2):
                    inconsistencies.append({
                        "type": "temporal_gap",
                        "severity": "warning" if gap < timedelta(hours=6) else "high",
                        "description": f"Gap of {gap} between events",
                        "between": [events[i]["event_id"], events[i + 1]["event_id"]],
                        "gap_hours": gap.total_seconds() / 3600,
                    })
                elif gap < timedelta(0):
                    inconsistencies.append({
                        "type": "chronological_error",
                        "severity": "high",
                        "description": f"Events appear out of chronological order",
                        "between": [events[i]["event_id"], events[i + 1]["event_id"]],
                        "gap_hours": gap.total_seconds() / 3600,
                    })

        # Check PRECEDES/FOLLOWS relationships match chronological order
        precedes_query = """
        MATCH (c:Case {case_id: $case_id})-[:HAPPENED_ON]->(t1:TimeEvent)-[:PRECEDES]->(t2:TimeEvent)
        RETURN properties(t1) as from_event, properties(t2) as to_event
        """
        precedes_results = await self._neo4j.execute_query(
            precedes_query, {"case_id": case_id}
        )
        for r in precedes_results:
            from_ts = self._parse_timestamp(
                r["from_event"].get("timestamp", "")
            )
            to_ts = self._parse_timestamp(
                r["to_event"].get("timestamp", "")
            )
            if from_ts and to_ts and from_ts > to_ts:
                inconsistencies.append({
                    "type": "relationship_mismatch",
                    "severity": "high",
                    "description": (
                        f"PRECEDES relationship but {r['from_event'].get('event_id')} "
                        f"has later timestamp than {r['to_event'].get('event_id')}"
                    ),
                    "between": [
                        r["from_event"].get("event_id"),
                        r["to_event"].get("event_id"),
                    ],
                })

        return inconsistencies

    async def reconstruct_timeline(self, case_id: str) -> Dict[str, Any]:
        """Full timeline reconstruction with gap analysis."""
        timeline = await self.get_case_timeline(case_id)
        inconsistencies = await self.detect_inconsistencies(case_id)

        events = timeline["events"]
        time_span = None
        gaps = []

        if len(events) >= 2:
            first_ts = self._parse_timestamp(events[0]["timestamp"])
            last_ts = self._parse_timestamp(events[-1]["timestamp"])
            if first_ts and last_ts:
                span = last_ts - first_ts
                time_span = {
                    "start": first_ts.isoformat(),
                    "end": last_ts.isoformat(),
                    "duration_hours": span.total_seconds() / 3600,
                }

            # Find gaps between consecutive events
            for i in range(len(events) - 1):
                ts1 = self._parse_timestamp(events[i]["timestamp"])
                ts2 = self._parse_timestamp(events[i + 1]["timestamp"])
                if ts1 and ts2:
                    gap_hours = (ts2 - ts1).total_seconds() / 3600
                    if gap_hours > 0.5:  # Flag gaps > 30 minutes
                        gaps.append({
                            "after_event": events[i]["event_id"],
                            "before_event": events[i + 1]["event_id"],
                            "gap_hours": round(gap_hours, 2),
                        })

        return {
            "case_id": case_id,
            "events": events,
            "total_events": len(events),
            "time_span": time_span,
            "gaps": gaps,
            "inconsistencies": inconsistencies,
        }
