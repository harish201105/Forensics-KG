#!/usr/bin/env python3
"""Graph-grounding ablation for the NL->Cypher reasoning layer.

Compares two conditions on the live knowledge graph (Neo4j), over a fixed set of
natural-language questions with known gold answers:

  * UNGROUNDED: naive LLM -> Cypher (no schema, no rules, no self-correction).
  * GROUNDED:   schema-grounded generation (the ontology schema + generation rules
                in the prompt) + a self-correction loop that feeds Neo4j errors back.

Metrics per condition: executable Cypher (runs without error), correct answer
(expected entity appears in the result rows), and hallucinated schema element
(query references a node label or relationship type that does not exist in the
graph). Writes query_ablation.json. Needs Neo4j running + OpenAI quota.
"""
import asyncio, json, re, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from neo4j import GraphDatabase
from app.config import get_settings
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.schemas import CYPHER_GENERATION_SCHEMA
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.query.nl_to_cypher import NLToCypherEngine

S = get_settings()

# (question, expected-substring-in-results) — answers grounded in the 28-case gold.
GOLD = [
    ("Who was the victim in the Tandoor murder case?", "naina sahni"),
    ("What weapon was used in the Tandoor murder case?", "revolver"),
    ("Which substance was used to kill the victims in the Koodathayi cyanide case?", "cyanide"),
    ("Who is the main accused in the murder of Shraddha Walkar?", "poonawalla"),
    ("Who was the victim in the Gauri Lankesh case?", "gauri lankesh"),
    ("What crime type is recorded for the Bilkis Bano case?", "rape"),
    ("Who was the victim in the Soumya murder case?", "soumya"),
    ("Which weapon is associated with the Murder of Jessica Lal?", "firearm"),
    ("Who was the victim in the Nirbhaya 2012 Delhi gang rape case?", "jyoti"),
    ("What crime type is recorded for the Murder of Narendra Dabholkar?", "murder"),
    ("Name an accused person in the Aarushi Talwar murder case.", "talwar"),
    ("Who is the accused in the murder of Pramod Mahajan?", "pravin"),
    ("Which person was the victim in the Sheena Bora case?", "sheena"),
    ("What weapon was used in the assassination of Phoolan Devi?", "firearm"),
    ("Who was the victim in the Priyadarshini Mattoo case?", "mattoo"),
    ("How many Case nodes are in the knowledge graph?", None),            # count
    ("List the crime types recorded across all cases.", "murder"),
    ("Which cases have a crime type of murder?", "murder"),
    ("Find persons whose role is victim across the cases.", "victim"),
    ("Which cases involve rape as a crime type?", "rape"),
    ("List people involved in the Kathua case.", None),
    ("What is the crime type of the R. G. Kar case?", "rape"),
    ("Who was the victim in the Jigisha Ghosh murder case?", "jigisha"),
    ("Which weapon type appears in the Pramod Mahajan case?", "firearm"),
]


def schema_elements(driver):
    with driver.session(database=S.neo4j_database) as s:
        labels = {r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label")}
        rels = {r["relationshipType"] for r in
                s.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")}
    return labels, rels


def cypher_labels_rels(q):
    labs = set(re.findall(r":\s*([A-Za-z_][A-Za-z0-9_]*)", q.split("RETURN")[0] if False else q))
    # labels appear as (:Label) or (x:Label); rels as [:REL] or [r:REL]
    labs = set(re.findall(r"\(\s*\w*\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", q))
    rels = set(re.findall(r"\[\s*\w*\s*:\s*([A-Za-z_][A-Za-z0-9_]*)", q))
    return labs, rels


def run_cypher(driver, q):
    try:
        with driver.session(database=S.neo4j_database) as s:
            rows = [r.data() for r in s.run(q)]
        return True, rows, ""
    except Exception as e:
        return False, None, str(e)[:160]


async def ungrounded_cypher(oc, question):
    sys_p = ("You are a Cypher expert. Generate a single Cypher query to answer the "
             "user's question about a forensics knowledge graph. Return only the query.")
    r = await oc.extract_structured(system_prompt=sys_p, user_prompt=question,
                                    response_schema=CYPHER_GENERATION_SCHEMA, temperature=0.0)
    return (r.get("cypher_query") or "").strip()


def hallucinated(cy, labels, rels):
    cl, cr = cypher_labels_rels(cy)
    bad_l = {x for x in cl if x not in labels}
    bad_r = {x for x in cr if x not in rels}
    return bool(bad_l or bad_r), (bad_l | bad_r)


def correct(rows, expected):
    if expected is None:                       # count/list questions: any non-empty result
        return bool(rows)
    blob = json.dumps(rows, default=str).lower()
    return expected.lower() in blob


async def main():
    driver = GraphDatabase.driver(S.neo4j_uri, auth=(S.neo4j_username, S.neo4j_password))
    labels, rels = schema_elements(driver)
    oc = OpenAIClient(S)
    nl = NLToCypherEngine(oc, ForensicsOntologySchema(S.ontology_path))

    agg = {c: {"exec": 0, "correct": 0, "halluc": 0} for c in ("ungrounded", "grounded")}
    rows_out = []
    for q, exp in GOLD:
        rec = {"q": q, "expected": exp}
        # --- ungrounded: single attempt ---
        ucy = await ungrounded_cypher(oc, q)
        uok, ures, uerr = run_cypher(driver, ucy)
        uh, _ = hallucinated(ucy, labels, rels)
        agg["ungrounded"]["exec"] += uok
        agg["ungrounded"]["correct"] += (uok and correct(ures, exp))
        agg["ungrounded"]["halluc"] += uh
        # --- grounded: schema prompt + self-correction (up to 3) ---
        gres = None
        gcy = (await nl.generate_cypher(q))["cypher_query"].strip()
        for attempt in range(3):
            gok, gres, gerr = run_cypher(driver, gcy)
            if gok:
                break
            fix = await nl.fix_cypher(q, gcy, gerr)
            nxt = (fix.get("cypher_query") or "").strip()
            if not nxt or nxt == gcy:
                break
            gcy = nxt
        gok = gres is not None
        gh, _ = hallucinated(gcy, labels, rels)
        agg["grounded"]["exec"] += gok
        agg["grounded"]["correct"] += (gok and correct(gres, exp))
        agg["grounded"]["halluc"] += gh
        rec.update({"ungrounded": {"exec": uok, "correct": uok and correct(ures, exp), "halluc": uh},
                    "grounded": {"exec": gok, "correct": gok and correct(gres, exp), "halluc": gh}})
        rows_out.append(rec)
        print(f"  U[e{int(uok)} c{int(uok and correct(ures,exp))} h{int(uh)}] "
              f"G[e{int(gok)} c{int(gok and correct(gres,exp))} h{int(gh)}]  {q[:54]}", flush=True)
    driver.close()

    n = len(GOLD)
    out = {"n_questions": n, "graph_labels": len(labels), "graph_rel_types": len(rels)}
    for c in ("ungrounded", "grounded"):
        out[c] = {k: round(agg[c][k] / n, 2) for k in ("exec", "correct", "halluc")}
        out[c]["counts"] = agg[c]
    out["per_question"] = rows_out
    (Path(__file__).parent / "query_ablation.json").write_text(json.dumps(out, indent=2))
    print("\n=== n =", n, "===")
    for c in ("ungrounded", "grounded"):
        e, co, h = out[c]["exec"], out[c]["correct"], out[c]["halluc"]
        print(f"{c:<11} executable={e}  correct={co}  hallucinated={h}")


if __name__ == "__main__":
    asyncio.run(main())
