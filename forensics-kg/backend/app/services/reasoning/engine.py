from typing import Dict, List, Any, Optional
import json
import numpy as np
from scipy import stats as scipy_stats
from loguru import logger
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.schemas import HYPOTHESIS_SCHEMA
from app.services.graph.neo4j_client import Neo4jClient


class ForensicReasoningEngine:
    """Forensic reasoning: statistical analysis + LLM hypothesis generation."""

    def __init__(self, openai_client: OpenAIClient, neo4j_client: Neo4jClient):
        self._openai = openai_client
        self._neo4j = neo4j_client

    async def generate_hypothesis(
        self, case_id: str, model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate hypotheses for a case based on all evidence in the KG."""
        # Step 1: Retrieve evidence
        evidence = await self._retrieve_case_evidence(case_id)
        if not evidence.get("found"):
            return {
                "primary_hypothesis": {
                    "hypothesis_id": "none",
                    "description": f"No data found for case {case_id}",
                    "mechanism": "unknown",
                    "confidence": 0.0,
                    "supporting_evidence": [],
                    "contradicting_evidence": [],
                    "likelihood_ratio": 0.0,
                },
                "alternative_hypotheses": [],
                "statistical_analysis": {},
                "recommendations": [f"Load data for case {case_id} first"],
                "reasoning_chain": ["No evidence available for analysis"],
                "confidence": 0.0,
            }

        # Step 2: Statistical analysis on stain features
        stat_analysis = self._perform_statistical_analysis(
            evidence.get("stain_features", [])
        )

        # Step 3: LLM hypothesis generation
        hypothesis_result = await self._generate_hypothesis_llm(
            evidence, stat_analysis, model_override=model_override
        )

        # Step 4: Store hypothesis in KG
        await self._store_hypothesis(case_id, hypothesis_result)

        return {
            "primary_hypothesis": hypothesis_result["primary_hypothesis"],
            "alternative_hypotheses": hypothesis_result["alternative_hypotheses"],
            "statistical_analysis": stat_analysis,
            "recommendations": hypothesis_result["recommendations"],
            "reasoning_chain": hypothesis_result["reasoning_chain"],
            "confidence": hypothesis_result["primary_hypothesis"]["confidence"],
        }

    def _perform_statistical_analysis(
        self, stain_features: List[Dict]
    ) -> Dict[str, Any]:
        """Statistical analysis (salvaged from reasoning_engine.py:229-298)."""
        if not stain_features:
            return {}

        feature_names = [
            "area", "perimeter", "aspect_ratio", "circularity", "orientation",
        ]
        numerical = {}
        for fname in feature_names:
            vals = [
                s.get(fname, 0)
                for s in stain_features
                if s.get(fname) is not None
            ]
            if vals:
                numerical[fname] = vals

        result = {
            "pattern_statistics": {},
            "feature_distributions": {},
            "confidence_intervals": {},
        }

        for feature, values in numerical.items():
            arr = np.array(values)
            result["pattern_statistics"][feature] = {
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "median": float(np.median(arr)),
                "count": len(values),
            }

            if len(values) > 3:
                try:
                    _, p_value = scipy_stats.shapiro(values[:5000])
                    mu, sigma = scipy_stats.norm.fit(values)
                    result["feature_distributions"][feature] = {
                        "mean": float(mu),
                        "std": float(sigma),
                        "normality_p_value": float(p_value),
                        "is_normal": p_value > 0.05,
                    }
                except Exception:
                    pass

            if len(values) > 1:
                try:
                    ci = scipy_stats.t.interval(
                        0.95,
                        len(values) - 1,
                        loc=np.mean(arr),
                        scale=scipy_stats.sem(arr),
                    )
                    result["confidence_intervals"][feature] = {
                        "lower": float(ci[0]),
                        "upper": float(ci[1]),
                        "level": 0.95,
                    }
                except Exception:
                    pass

        # Correlation matrix if enough features
        if len(numerical) >= 2:
            common_len = min(len(v) for v in numerical.values())
            if common_len > 2:
                matrix_data = {
                    k: v[:common_len] for k, v in numerical.items()
                }
                keys = list(matrix_data.keys())
                matrix = np.corrcoef(
                    [matrix_data[k] for k in keys]
                )
                result["correlation_matrix"] = {
                    "features": keys,
                    "values": matrix.tolist(),
                }

        return result

    async def _retrieve_case_evidence(self, case_id: str) -> Dict[str, Any]:
        """Retrieve all evidence for a case from Neo4j."""
        # Try case first
        query = """
        MATCH (c:Case {case_id: $case_id})
        OPTIONAL MATCH (c)-[:HAS_EVIDENCE]->(e:Evidence)
        OPTIONAL MATCH (c)<-[:INVOLVED_IN]-(p:Person)
        OPTIONAL MATCH (c)-[:OCCURRED_AT]->(l:Location)
        OPTIONAL MATCH (c)-[:HAS_PATTERN]->(bp:BloodstainPattern)
        OPTIONAL MATCH (bp)-[:CONTAINS_STAIN]->(s:Stain)
        OPTIONAL MATCH (bp)-[:GENERATED_BY]->(m:ImpactMechanismNode)
        RETURN c as case_data,
               collect(DISTINCT properties(e)) as evidence,
               collect(DISTINCT properties(p)) as persons,
               collect(DISTINCT properties(l)) as locations,
               collect(DISTINCT properties(bp)) as patterns,
               collect(DISTINCT properties(s)) as stains,
               collect(DISTINCT properties(m)) as mechanisms
        """
        results = await self._neo4j.execute_query(query, {"case_id": case_id})

        if not results or not results[0].get("case_data"):
            # Try as experiment
            exp_query = """
            MATCH (exp:Experiment {experiment_id: $case_id})
            OPTIONAL MATCH (exp)<-[:CAUSED_BY]-(bp:BloodstainPattern)
            OPTIONAL MATCH (bp)-[:CONTAINS_STAIN]->(s:Stain)
            OPTIONAL MATCH (bp)-[:GENERATED_BY]->(m:ImpactMechanismNode)
            RETURN exp as case_data,
                   collect(DISTINCT properties(bp)) as patterns,
                   collect(DISTINCT properties(s)) as stains,
                   collect(DISTINCT properties(m)) as mechanisms
            """
            results = await self._neo4j.execute_query(
                exp_query, {"case_id": case_id}
            )

        if not results or not results[0].get("case_data"):
            return {"found": False}

        r = results[0]
        return {
            "found": True,
            "case": r.get("case_data", {}),
            "evidence": r.get("evidence", []),
            "persons": r.get("persons", []),
            "locations": r.get("locations", []),
            "patterns": r.get("patterns", []),
            "stain_features": r.get("stains", []),
            "mechanisms": r.get("mechanisms", []),
        }

    async def _generate_hypothesis_llm(
        self, evidence: Dict, stat_analysis: Dict, model_override: Optional[str] = None
    ) -> Dict[str, Any]:
        system_prompt = (
            "You are a forensic scientist specializing in crime scene "
            "reconstruction and bloodstain pattern analysis. Analyze the "
            "evidence and generate hypotheses about what happened. "
            "Be scientifically rigorous and cite specific evidence."
        )
        # Truncate for token limits
        evidence_str = json.dumps(
            {k: v for k, v in evidence.items() if k != "stain_features"},
            indent=2, default=str,
        )[:3000]
        stats_str = json.dumps(stat_analysis, indent=2)[:2000]

        user_prompt = (
            f"Evidence summary:\n{evidence_str}\n\n"
            f"Statistical analysis of features:\n{stats_str}\n\n"
            f"Generate forensic hypotheses:\n"
            f"1. Primary hypothesis with detailed reasoning\n"
            f"2. 2-3 alternative hypotheses\n"
            f"3. Specific recommendations for further investigation\n"
            f"4. Step-by-step reasoning chain"
        )
        return await self._openai.extract_structured(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_schema=HYPOTHESIS_SCHEMA,
            model_override=model_override,
        )

    async def _store_hypothesis(
        self, case_id: str, hypothesis: Dict
    ) -> None:
        """Store generated hypothesis in the KG."""
        try:
            primary = hypothesis["primary_hypothesis"]
            await self._neo4j.merge_node(
                "Hypothesis",
                {"hypothesis_id": primary["hypothesis_id"]},
                {
                    "description": primary["description"],
                    "mechanism": primary["mechanism"],
                    "confidence": primary["confidence"],
                    "supporting_evidence": json.dumps(
                        primary.get("supporting_evidence", [])
                    ),
                    "contradicting_evidence": json.dumps(
                        primary.get("contradicting_evidence", [])
                    ),
                    "likelihood_ratio": primary.get("likelihood_ratio", 0),
                },
            )
            # Link to case
            await self._neo4j.merge_relationship(
                source_label="Case",
                source_key="case_id",
                source_value=case_id,
                target_label="Hypothesis",
                target_key="hypothesis_id",
                target_value=primary["hypothesis_id"],
                rel_type="HAS_EVIDENCE",
                properties={"relevance": primary["confidence"]},
            )
        except Exception as e:
            logger.warning(f"Could not store hypothesis in KG: {e}")
