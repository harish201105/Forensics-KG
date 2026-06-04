"""Cross-case entity resolution.

Each document/case extraction creates its own island of Person/Location/Weapon
nodes, so the same real-world entity (a recurring judge, a shared location, a
weapon type) appears as separate nodes per case. This layer links those
duplicates with non-destructive ``SAME_AS`` edges so cross-case queries
("show all cases involving X") actually return connected results.

Method (hybrid entity resolution):
  1. BLOCKING — use the node embeddings (cosine) to generate candidate pairs,
     so we only compare semantically-near nodes instead of all O(n^2).
  2. DECISION — confirm a pair with a type-specific string match (normalized
     name/type similarity) so we never merge two genuinely different people who
     merely share a role. Forensic false-merges are dangerous, so thresholds
     are conservative and provenance is preserved (we link, never delete).
"""

import re
from difflib import SequenceMatcher
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from app.services.graph.neo4j_client import Neo4jClient

# Labels we resolve, and the property that names/identifies them.
RESOLVABLE: Dict[str, str] = {
    "Person": "name",
    "Location": "name",
    "Weapon": "type",
    "Vehicle": "type",
    "CrimeType": "name",
}
# Titles / honorifics / role words stripped before comparing person names.
_PERSON_NOISE = re.compile(
    r"\b(j|jj|mr|mrs|ms|dr|md|smt|sri|shri|justice|hon'?ble|inspector|"
    r"sub-?inspector|si|asi|constable|the|late|deceased|accused|appellant|"
    r"respondent|petitioner|plaintiff|defendant)\b",
    re.IGNORECASE,
)
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")
# Generic placeholder values that must never be treated as a shared entity.
_GENERIC = frozenset({"weapon", "vehicle", "unknown", "n/a", "na", "none", "person",
                      "victim", "suspect", "accused", "object", "item", ""})


def _normalize(label: str, value: str) -> str:
    v = (value or "").strip().lower()
    if label == "Person":
        v = _PERSON_NOISE.sub(" ", v)
    v = _NON_ALNUM.sub(" ", v)
    return " ".join(v.split())


# Roles that legitimately recur across cases with no shared context (judges,
# investigating officers, forensic analysts) — name+role is the identifying signal.
_DISTINCTIVE_ROLES = frozenset({"officer", "forensic_analyst", "judge", "examiner"})


def _name_score(norm_a: str, norm_b: str):
    """Return (score, min_tokens, subset_used, initials_conflict)."""
    base = SequenceMatcher(None, norm_a, norm_b).ratio()
    ta, tb = norm_a.split(), norm_b.split()
    sa, sb = set(ta), set(tb)
    subset = bool(sa and sb and (sa <= sb or sb <= sa))
    # Subset bonus ONLY when the shorter name has >=2 tokens — prevents merging a
    # bare common first name into a fuller name (e.g. "anjali" ⊄ "anjali mehra").
    if subset and min(len(ta), len(tb)) >= 2:
        base = max(base, 0.95)
    # Conflicting initials: short tokens (<=2 chars) present in both but disjoint —
    # e.g. "g c mathur" vs "g s mathur" string-match ~0.9 but are different people.
    short_a = {t for t in sa if len(t) <= 2}
    short_b = {t for t in sb if len(t) <= 2}
    conflict = bool(short_a and short_b and not (short_a <= short_b or short_b <= short_a))
    return base, min(len(ta), len(tb)), subset, conflict


def _decide_match(name_score, min_tokens, conflict, role_a, role_b, ctx_overlap, name_threshold):
    """Context-aware accept/reject + a transparent confidence score.

    Strong, unambiguous multi-token names pass on the name alone. Weak names
    (single-token / common / conflicting initials) require corroboration: shared
    co-occurring entities OR a distinctive recurring role (judge/officer)."""
    if name_score < name_threshold:
        return False, 0.0, ""
    role_match = bool(role_a and role_b and role_a == role_b)
    distinctive = role_match and role_a in _DISTINCTIVE_ROLES
    ambiguous = (min_tokens < 2) or conflict
    if ambiguous and not (distinctive or ctx_overlap > 0):
        reason = "conflicting-initials" if conflict else "ambiguous-name"
        return False, 0.0, reason + "-no-corroboration"
    confidence = round(0.6 * name_score + 0.25 * min(ctx_overlap, 1.0)
                       + 0.15 * (1.0 if role_match else 0.0), 3)
    basis = "name"
    if ctx_overlap > 0:
        basis += "+context"
    if distinctive:
        basis += "+role"
    return True, confidence, basis


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b) if (a | b) else 0.0


def _doc_prefix(node_id: str) -> str:
    """The source-document/case proxy. Entity IDs are '<docid>_<type>_<n>', where
    <docid> may contain hyphens/underscores (e.g. 'doc_16bead11', 'FIR-2024-0012').
    Strip the trailing '_<type>_<number>' to recover the document id:
      'doc_16bead11_person_1'   -> 'doc_16bead11'
      'FIR-2024-0012_person_1'  -> 'FIR-2024-0012'
    """
    if not node_id:
        return ""
    parts = node_id.split("_")
    if len(parts) >= 3 and parts[-1].isdigit():
        return "_".join(parts[:-2])
    return node_id


class _UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def find(self, x: int) -> int:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


class EntityResolutionService:
    """Links duplicate entities across cases via embedding-blocked, name-confirmed
    SAME_AS relationships."""

    def __init__(self, neo4j_client: Neo4jClient):
        self._neo4j = neo4j_client

    async def _fetch_nodes(self, label: str, prop: str) -> List[Dict[str, Any]]:
        """Fetch resolvable nodes with their value, role, doc, embedding, and the
        set of co-occurring entity names (2-hop neighbours, excluding Case & SAME_AS)."""
        rows = await self._neo4j.execute_query(
            f"""
            MATCH (n:{label})
            WHERE n.embedding IS NOT NULL AND n.{prop} IS NOT NULL
            OPTIONAL MATCH (n)-[rels*1..2]-(m)
            WHERE m <> n
              AND none(rr IN rels WHERE type(rr) = 'SAME_AS')
              AND none(l IN labels(m) WHERE l IN ['Case', 'Embedded'])
            WITH n, collect(DISTINCT toLower(coalesce(m.name, m.title, '')))[..40] AS ctx
            RETURN elementId(n) AS eid, n.{prop} AS val, n.role AS role,
                   [k IN keys(n) WHERE k ENDS WITH '_id' | n[k]][0] AS id,
                   n.embedding AS emb, [c IN ctx WHERE c <> ''] AS ctx
            """
        )
        items = []
        for r in rows:
            norm = _normalize(label, r.get("val", ""))
            if not norm or norm in _GENERIC:
                continue
            items.append({
                "eid": r["eid"], "val": r["val"], "norm": norm,
                "role": (r.get("role") or "").lower(),
                "doc": _doc_prefix(r.get("id") or ""),
                "emb": r["emb"], "ctx": set(r.get("ctx") or []),
            })
        return items

    async def resolve(
        self,
        labels: Optional[List[str]] = None,
        name_threshold: float = 0.88,
        candidate_floor: float = 0.78,
        cross_case_only: bool = True,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        import numpy as np

        labels = labels or list(RESOLVABLE.keys())
        summary: Dict[str, Any] = {"labels": {}, "links_created": 0, "dry_run": dry_run}

        for label in labels:
            prop = RESOLVABLE.get(label)
            if not prop:
                continue
            items = await self._fetch_nodes(label, prop)
            n = len(items)
            if n < 2:
                summary["labels"][label] = {"nodes": n, "pairs_linked": 0, "clusters": 0}
                continue

            M = np.asarray([it["emb"] for it in items], dtype=float)
            norms = np.linalg.norm(M, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            M = M / norms
            sim = M @ M.T  # cosine

            uf = _UnionFind()
            pairs: List[Tuple[int, int, float, str]] = []
            for i in range(n):
                for j in range(i + 1, n):
                    if cross_case_only and items[i]["doc"] == items[j]["doc"]:
                        continue
                    if float(sim[i, j]) < candidate_floor:
                        continue
                    name_score, min_tokens, _, conflict = _name_score(items[i]["norm"], items[j]["norm"])
                    ctx_overlap = _jaccard(items[i]["ctx"], items[j]["ctx"])
                    accept, conf, basis = _decide_match(
                        name_score, min_tokens, conflict, items[i]["role"], items[j]["role"],
                        ctx_overlap, name_threshold,
                    )
                    if accept:
                        pairs.append((i, j, conf, basis))
                        uf.union(i, j)

            # Create SAME_AS edges (clique within each matched pair set).
            if not dry_run:
                for i, j, conf, basis in pairs:
                    await self._neo4j.execute_query(
                        """
                        MATCH (a) WHERE elementId(a) = $a
                        MATCH (b) WHERE elementId(b) = $b
                        MERGE (a)-[r:SAME_AS]->(b)
                        SET r.score = $score, r.basis = $basis
                        """,
                        {"a": items[i]["eid"], "b": items[j]["eid"],
                         "score": conf, "basis": f"{basis} ({RESOLVABLE[label]})"},
                    )

            clusters: Dict[int, int] = {}
            for idx in range(n):
                if idx in uf.parent:
                    clusters[uf.find(idx)] = clusters.get(uf.find(idx), 0) + 1
            multi = {k: c for k, c in clusters.items() if c > 1}
            summary["labels"][label] = {
                "nodes": n,
                "pairs_linked": len(pairs),
                "clusters": len(multi),
                "examples": self._examples(items, pairs)[:5],
            }
            summary["links_created"] += len(pairs)
            logger.info(
                f"Entity resolution [{label}]: {len(pairs)} SAME_AS links, "
                f"{len(multi)} cross-case clusters"
            )

        return summary

    async def resolve_incremental(
        self,
        new_doc_prefix: str,
        labels: Optional[List[str]] = None,
        name_threshold: float = 0.88,
        candidate_floor: float = 0.78,
    ) -> Dict[str, Any]:
        """Resolve only the entities just added (matching new_doc_prefix) against
        the rest of the graph. O(new x total) — cheap enough to run after each
        ingestion so the graph stays connected as cases are added."""
        import numpy as np

        labels = labels or list(RESOLVABLE.keys())
        created = 0
        for label in labels:
            prop = RESOLVABLE.get(label)
            if not prop:
                continue
            items = await self._fetch_nodes(label, prop)
            new_idx = [i for i, it in enumerate(items) if it["doc"] == new_doc_prefix]
            if not new_idx or len(items) < 2:
                continue
            M = np.asarray([it["emb"] for it in items], dtype=float)
            nrm = np.linalg.norm(M, axis=1, keepdims=True)
            nrm[nrm == 0] = 1.0
            M = M / nrm
            for i in new_idx:
                sims = M[i] @ M.T
                for j in range(len(items)):
                    if j == i or items[j]["doc"] == new_doc_prefix:
                        continue
                    if float(sims[j]) < candidate_floor:
                        continue
                    name_score, min_tokens, _, conflict = _name_score(items[i]["norm"], items[j]["norm"])
                    ctx_overlap = _jaccard(items[i]["ctx"], items[j]["ctx"])
                    accept, conf, basis = _decide_match(
                        name_score, min_tokens, conflict, items[i]["role"], items[j]["role"],
                        ctx_overlap, name_threshold,
                    )
                    if accept:
                        await self._neo4j.execute_query(
                            """
                            MATCH (a) WHERE elementId(a) = $a
                            MATCH (b) WHERE elementId(b) = $b
                            MERGE (a)-[r:SAME_AS]->(b)
                            SET r.score = $score, r.basis = $basis
                            """,
                            {"a": items[i]["eid"], "b": items[j]["eid"],
                             "score": conf, "basis": f"{basis} ({prop}, incremental)"},
                        )
                        created += 1
        if created:
            logger.info(f"Incremental resolution: {created} SAME_AS link(s) for {new_doc_prefix}")
        return {"links_created": created, "doc": new_doc_prefix}

    @staticmethod
    def _examples(items, pairs) -> List[str]:
        seen, out = set(), []
        for i, j, conf, basis in pairs:
            key = tuple(sorted((items[i]["norm"], items[j]["norm"])))
            if key in seen:
                continue
            seen.add(key)
            out.append(f"'{items[i]['val']}' == '{items[j]['val']}' "
                       f"(conf {conf:.2f}, {basis})")
        return out

    async def cross_case(
        self, label: str, value: str, max_hops: int = 2
    ) -> Dict[str, Any]:
        """All cases connected to an entity, following SAME_AS across cases."""
        prop = RESOLVABLE.get(label, "name")
        cypher = f"""
        MATCH (seed:{label})
        WHERE toLower(seed.{prop}) = toLower($value)
        MATCH (seed)-[:SAME_AS*0..3]-(syn:{label})
        OPTIONAL MATCH (syn)-[*1..{max_hops}]-(c:Case)
        WITH collect(DISTINCT syn.{prop}) AS aliases,
             collect(DISTINCT c.case_id) AS case_ids,
             collect(DISTINCT c.title) AS titles
        RETURN aliases, [x IN case_ids WHERE x IS NOT NULL] AS case_ids,
               [t IN titles WHERE t IS NOT NULL] AS titles
        """
        rows = await self._neo4j.execute_query(cypher, {"value": value})
        if not rows:
            return {"entity": value, "label": label, "cases": [], "aliases": []}
        r = rows[0]
        cases = sorted(set(r.get("case_ids", [])))
        return {
            "entity": value,
            "label": label,
            "aliases": r.get("aliases", []),
            "case_count": len(cases),
            "cases": cases,
            "case_titles": r.get("titles", []),
        }

    async def count_same_as(self) -> int:
        rows = await self._neo4j.execute_query(
            "MATCH ()-[r:SAME_AS]->() RETURN count(r) AS c"
        )
        return rows[0]["c"] if rows else 0
