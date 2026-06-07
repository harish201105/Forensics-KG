#!/usr/bin/env python3
"""Ingest the 28 documented real cases into Neo4j so the graph-grounding NL->Cypher
ablation can be run against the paper's own data. Entity ids are namespaced per case
(RC-xxx__person_1) to avoid MERGE collisions across cases and with existing data.
Idempotent (MERGE). Needs Neo4j running + OpenAI quota."""
import asyncio, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from app.config import get_settings
from app.services.extraction.openai_client import OpenAIClient
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.text_extractor import TextExtractor
from app.services.graph.neo4j_client import Neo4jClient
from app.services.graph.operations import GraphOperations

S = get_settings()


async def main():
    oc = OpenAIClient(S)
    schema = ForensicsOntologySchema(S.ontology_path)
    te = TextExtractor(oc, schema)
    client = Neo4jClient(S.neo4j_uri, S.neo4j_username, S.neo4j_password, S.neo4j_database)
    await client.connect()
    ops = GraphOperations(client, schema)

    cases = [json.loads(p.read_text()) for p in sorted((Path(S.data_dir) / "real_cases").glob("*.json"))]
    tot_e = tot_r = 0
    for c in cases:
        cid = c["case_id"]
        r = await te.extract(c["narrative"], "fir")
        ents, rels = r.get("entities", []), r.get("relationships", [])
        for e in ents:
            e["entity_id"] = f"{cid}__{e['entity_id']}"
            props = e.setdefault("properties", {})
            props.pop(ops._get_id_key(e["entity_type"]), None)   # avoid overwriting merge key
            props["source_case"] = cid
            if e["entity_type"] == "Case":
                props["title"] = c.get("title", props.get("title", cid))
        for x in rels:
            x["source_entity_id"] = f"{cid}__{x['source_entity_id']}"
            x["target_entity_id"] = f"{cid}__{x['target_entity_id']}"
        se = await ops.store_extracted_entities(ents)
        sr = await ops.store_extracted_relationships(rels, ents)
        tot_e += len(se); tot_r += sr
        print(f"  {cid:<22} entities={len(se)} rels={sr}", flush=True)

    # CASE-ANCHORING normalization: ensure every extracted entity is connected to
    # its Case via the ontology-defined relationship, so case-scoped queries are a
    # uniform 1 hop (LLM extraction otherwise links some entities only via events).
    anchors = [("(p:Person)", "(p)-[:INVOLVED_IN]->(c)"),
               ("(l:Location)", "(c)-[:OCCURRED_AT]->(l)"),
               ("(x:CrimeType)", "(c)-[:HAS_CRIME_TYPE]->(x)"),
               ("(t:TimeEvent)", "(c)-[:HAPPENED_ON]->(t)"),
               ("(e:Evidence)", "(c)-[:HAS_EVIDENCE]->(e)")]
    for mp, merge in anchors:
        var = mp[1]
        await client.execute_write(
            f"MATCH (c:Case) WHERE c.source_case STARTS WITH 'RC-' "
            f"MATCH {mp} WHERE {var}.source_case = c.source_case MERGE {merge}")
    print("case-anchoring normalization applied")
    await client.disconnect()
    print(f"\ningested {len(cases)} cases: {tot_e} entities, {tot_r} relationships")


if __name__ == "__main__":
    asyncio.run(main())
