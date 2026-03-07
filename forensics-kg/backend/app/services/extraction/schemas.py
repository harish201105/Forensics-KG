"""JSON Schemas for OpenAI structured output responses."""

# --- FIR Text Extraction Schemas (two-stage) ---

FIR_ENTITY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "Case", "Person", "Location", "Evidence",
                            "Weapon", "Vehicle", "CrimeType", "TimeEvent",
                        ],
                    },
                    "entity_id": {"type": "string"},
                    "properties": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "status": {"type": "string"},
                            "crime_type": {"type": "string"},
                            "date": {"type": "string"},
                            "location": {"type": "string"},
                            "type": {"type": "string"},
                            "value": {"type": "string"},
                        },
                        "required": ["name", "description", "role", "status", "crime_type", "date", "location", "type", "value"],
                        "additionalProperties": False,
                    },
                },
                "required": ["entity_type", "entity_id", "properties"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}

FIR_RELATIONSHIP_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_entity_id": {"type": "string"},
                    "target_entity_id": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": [
                            "INVOLVED_IN", "OCCURRED_AT", "USED_WEAPON",
                            "HAS_EVIDENCE", "HAPPENED_ON", "RELATED_TO_CASE",
                            "WITNESSED_BY", "ARRESTED_BY", "SUPPORTS",
                            "CONTRADICTS", "PRECEDES", "FOLLOWS",
                            "FOUND_AT", "BELONGS_TO", "HAS_CRIME_TYPE",
                            "HAS_PATTERN",
                        ],
                    },
                    "properties": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "weight": {"type": "string"},
                        },
                        "required": ["description", "role", "weight"],
                        "additionalProperties": False,
                    },
                    "confidence": {"type": "number"},
                    "evidence_span": {"type": "string"},
                },
                "required": [
                    "source_entity_id", "target_entity_id",
                    "relationship_type", "properties", "confidence", "evidence_span",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["relationships"],
    "additionalProperties": False,
}

# --- Bloodstain Image Analysis Schema ---

BLOODSTAIN_IMAGE_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern_type": {
            "type": "string",
            "enum": [
                "passive_drop", "transfer", "projected", "cast_off",
                "impact_spatter", "arterial_spurt", "void", "wipe",
                "swipe", "flow", "pool", "satellite",
            ],
        },
        "impact_mechanism": {
            "type": "string",
            "enum": [
                "blunt_force", "sharp_force", "gunshot",
                "stabbing", "beating", "fall", "unknown",
            ],
        },
        "confidence": {"type": "number"},
        "description": {"type": "string"},
        "key_features": {"type": "array", "items": {"type": "string"}},
        "spatial_distribution": {"type": "string"},
        "estimated_stain_count": {"type": "integer"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "pattern_type", "impact_mechanism", "confidence",
        "description", "key_features", "spatial_distribution",
        "estimated_stain_count", "uncertainties",
    ],
    "additionalProperties": False,
}

# --- Bloodstain Combined Analysis Schema (CV features + vision + metadata) ---

BLOODSTAIN_COMBINED_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern_type": {"type": "string"},
        "impact_mechanism": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "key_features": {"type": "array", "items": {"type": "string"}},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "from_entity": {"type": "string"},
                    "to_entity": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["type", "from_entity", "to_entity", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
    },
    "required": [
        "pattern_type", "impact_mechanism", "confidence",
        "reasoning", "key_features", "uncertainties", "relationships",
    ],
    "additionalProperties": False,
}

# --- Synthetic FIR Generation Schema ---

SYNTHETIC_FIR_SCHEMA = {
    "type": "object",
    "properties": {
        "fir_number": {"type": "string"},
        "date_filed": {"type": "string"},
        "police_station": {"type": "string"},
        "crime_type": {"type": "string"},
        "narrative": {"type": "string"},
        "complainant_name": {"type": "string"},
        "suspect_description": {"type": "string"},
        "victim_name": {"type": "string"},
        "victim_description": {"type": "string"},
        "location_details": {"type": "string"},
        "evidence_description": {"type": "string"},
        "bloodstain_evidence": {"type": "string"},
        "weapon_details": {"type": "string"},
        "vehicle_details": {"type": "string"},
        "witness_statements": {"type": "array", "items": {"type": "string"}},
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
    },
    "required": [
        "fir_number", "date_filed", "police_station", "crime_type",
        "narrative", "complainant_name", "location_details",
        "evidence_description", "witness_statements", "timeline_events",
        "suspect_description", "victim_name", "victim_description",
        "bloodstain_evidence", "weapon_details", "vehicle_details",
    ],
    "additionalProperties": False,
}

# --- NL-to-Cypher Schema ---

CYPHER_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "cypher_query": {"type": "string"},
        "explanation": {"type": "string"},
        "expected_result_type": {
            "type": "string",
            "enum": ["nodes", "relationships", "paths", "aggregation", "subgraph"],
        },
    },
    "required": ["cypher_query", "explanation", "expected_result_type"],
    "additionalProperties": False,
}

ANSWER_GENERATION_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "key_entities": {"type": "array", "items": {"type": "string"}},
        "follow_up_questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["answer", "confidence", "reasoning", "key_entities", "follow_up_questions"],
    "additionalProperties": False,
}

# --- Hypothesis Generation Schema ---

HYPOTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_hypothesis": {
            "type": "object",
            "properties": {
                "hypothesis_id": {"type": "string"},
                "description": {"type": "string"},
                "mechanism": {"type": "string"},
                "confidence": {"type": "number"},
                "supporting_evidence": {"type": "array", "items": {"type": "string"}},
                "contradicting_evidence": {"type": "array", "items": {"type": "string"}},
                "likelihood_ratio": {"type": "number"},
            },
            "required": [
                "hypothesis_id", "description", "mechanism", "confidence",
                "supporting_evidence", "contradicting_evidence", "likelihood_ratio",
            ],
            "additionalProperties": False,
        },
        "alternative_hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "hypothesis_id": {"type": "string"},
                    "description": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["hypothesis_id", "description", "mechanism", "confidence"],
                "additionalProperties": False,
            },
        },
        "recommendations": {"type": "array", "items": {"type": "string"}},
        "reasoning_chain": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "primary_hypothesis", "alternative_hypotheses",
        "recommendations", "reasoning_chain",
    ],
    "additionalProperties": False,
}

# --- Court Judgment Entity Extraction Schema ---

COURT_JUDGMENT_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "Case", "Person", "Location", "Evidence",
                            "Weapon", "Vehicle", "CrimeType", "TimeEvent",
                            "LegalSection", "Verdict", "CourtOrder",
                        ],
                    },
                    "entity_id": {"type": "string"},
                    "properties": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "status": {"type": "string"},
                            "crime_type": {"type": "string"},
                            "date": {"type": "string"},
                            "location": {"type": "string"},
                            "type": {"type": "string"},
                            "value": {"type": "string"},
                            "section_number": {"type": "string"},
                            "act_name": {"type": "string"},
                            "outcome": {"type": "string"},
                            "sentence": {"type": "string"},
                            "order_type": {"type": "string"},
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                },
                "required": ["entity_type", "entity_id", "properties"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}

COURT_JUDGMENT_RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_entity_id": {"type": "string"},
                    "target_entity_id": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": [
                            "INVOLVED_IN", "OCCURRED_AT", "USED_WEAPON",
                            "HAS_EVIDENCE", "HAPPENED_ON", "RELATED_TO_CASE",
                            "WITNESSED_BY", "ARRESTED_BY", "SUPPORTS",
                            "CONTRADICTS", "PRECEDES", "FOLLOWS",
                            "FOUND_AT", "BELONGS_TO", "HAS_CRIME_TYPE",
                            "PRESIDED_BY", "REPRESENTED_BY", "RESULTED_IN",
                            "CHARGED_UNDER", "ORDERED",
                        ],
                    },
                    "properties": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "weight": {"type": "string"},
                        },
                        "required": ["description"],
                        "additionalProperties": False,
                    },
                    "confidence": {"type": "number"},
                    "evidence_span": {"type": "string"},
                },
                "required": [
                    "source_entity_id", "target_entity_id",
                    "relationship_type", "properties", "confidence", "evidence_span",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["relationships"],
    "additionalProperties": False,
}

# --- Post-Mortem Report Entity Extraction Schema ---

POSTMORTEM_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "Case", "Person", "Location", "Evidence",
                            "TimeEvent", "InjuryPattern", "CauseOfDeath",
                            "ToxicologyResult", "OrganFinding",
                        ],
                    },
                    "entity_id": {"type": "string"},
                    "properties": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "type": {"type": "string"},
                            "injury_type": {"type": "string"},
                            "anatomical_location": {"type": "string"},
                            "dimensions": {"type": "string"},
                            "injury_age": {"type": "string"},
                            "primary_cause": {"type": "string"},
                            "contributing_factors": {"type": "string"},
                            "substance": {"type": "string"},
                            "concentration": {"type": "string"},
                            "method": {"type": "string"},
                            "organ": {"type": "string"},
                            "condition": {"type": "string"},
                            "date": {"type": "string"},
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                },
                "required": ["entity_type", "entity_id", "properties"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}

POSTMORTEM_RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_entity_id": {"type": "string"},
                    "target_entity_id": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": [
                            "INVOLVED_IN", "HAS_EVIDENCE", "HAPPENED_ON",
                            "FOUND_AT", "SUPPORTS", "CONTRADICTS",
                            "PRECEDES", "FOLLOWS", "INJURY_CAUSED_BY",
                            "LED_TO_DEATH", "TOXICOLOGY_OF", "ORGAN_FINDING_OF",
                        ],
                    },
                    "properties": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "weight": {"type": "string"},
                        },
                        "required": ["description"],
                        "additionalProperties": False,
                    },
                    "confidence": {"type": "number"},
                    "evidence_span": {"type": "string"},
                },
                "required": [
                    "source_entity_id", "target_entity_id",
                    "relationship_type", "properties", "confidence", "evidence_span",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["relationships"],
    "additionalProperties": False,
}

# --- Forensic Lab Report Entity Extraction Schema ---

LAB_REPORT_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "Case", "Person", "Evidence", "TimeEvent",
                            "Sample", "TestResult", "DNAProfile", "ChemicalCompound",
                        ],
                    },
                    "entity_id": {"type": "string"},
                    "properties": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "type": {"type": "string"},
                            "sample_type": {"type": "string"},
                            "collection_date": {"type": "string"},
                            "source": {"type": "string"},
                            "test_name": {"type": "string"},
                            "result": {"type": "string"},
                            "method": {"type": "string"},
                            "loci": {"type": "string"},
                            "alleles": {"type": "string"},
                            "match_probability": {"type": "string"},
                            "compound_name": {"type": "string"},
                            "concentration": {"type": "string"},
                            "date": {"type": "string"},
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                },
                "required": ["entity_type", "entity_id", "properties"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}

LAB_REPORT_RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_entity_id": {"type": "string"},
                    "target_entity_id": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": [
                            "HAS_EVIDENCE", "HAPPENED_ON", "SUPPORTS",
                            "CONTRADICTS", "TESTED_FROM", "RESULTED_IN",
                            "MATCHES_DNA", "CONTAINS_COMPOUND", "COLLECTED_BY",
                        ],
                    },
                    "properties": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "weight": {"type": "string"},
                        },
                        "required": ["description"],
                        "additionalProperties": False,
                    },
                    "confidence": {"type": "number"},
                    "evidence_span": {"type": "string"},
                },
                "required": [
                    "source_entity_id", "target_entity_id",
                    "relationship_type", "properties", "confidence", "evidence_span",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["relationships"],
    "additionalProperties": False,
}

# --- Witness Deposition Entity Extraction Schema ---

DEPOSITION_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "Case", "Person", "Location", "Evidence",
                            "TimeEvent", "Statement",
                        ],
                    },
                    "entity_id": {"type": "string"},
                    "properties": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "type": {"type": "string"},
                            "content": {"type": "string"},
                            "context": {"type": "string"},
                            "statement_type": {"type": "string"},
                            "date": {"type": "string"},
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                },
                "required": ["entity_type", "entity_id", "properties"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}

DEPOSITION_RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_entity_id": {"type": "string"},
                    "target_entity_id": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": [
                            "INVOLVED_IN", "OCCURRED_AT", "HAS_EVIDENCE",
                            "HAPPENED_ON", "WITNESSED_BY", "SUPPORTS",
                            "CONTRADICTS", "PRECEDES", "FOLLOWS",
                            "STATED_BY", "CORROBORATES",
                        ],
                    },
                    "properties": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "weight": {"type": "string"},
                        },
                        "required": ["description"],
                        "additionalProperties": False,
                    },
                    "confidence": {"type": "number"},
                    "evidence_span": {"type": "string"},
                },
                "required": [
                    "source_entity_id", "target_entity_id",
                    "relationship_type", "properties", "confidence", "evidence_span",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["relationships"],
    "additionalProperties": False,
}

# --- SOCO Report Entity Extraction Schema ---

SOCO_ENTITY_SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {
                        "type": "string",
                        "enum": [
                            "Case", "Person", "Location", "Evidence",
                            "TimeEvent", "CrimeScene", "PhysicalEvidence",
                            "ChainOfCustody",
                        ],
                    },
                    "entity_id": {"type": "string"},
                    "properties": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "type": {"type": "string"},
                            "scene_type": {"type": "string"},
                            "dimensions": {"type": "string"},
                            "conditions": {"type": "string"},
                            "item_description": {"type": "string"},
                            "location_in_scene": {"type": "string"},
                            "collection_method": {"type": "string"},
                            "handler": {"type": "string"},
                            "action": {"type": "string"},
                            "date": {"type": "string"},
                        },
                        "required": ["name", "description"],
                        "additionalProperties": False,
                    },
                },
                "required": ["entity_type", "entity_id", "properties"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["entities"],
    "additionalProperties": False,
}

SOCO_RELATIONSHIP_SCHEMA = {
    "type": "object",
    "properties": {
        "relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_entity_id": {"type": "string"},
                    "target_entity_id": {"type": "string"},
                    "relationship_type": {
                        "type": "string",
                        "enum": [
                            "INVOLVED_IN", "OCCURRED_AT", "HAS_EVIDENCE",
                            "HAPPENED_ON", "FOUND_AT", "SUPPORTS",
                            "PRECEDES", "FOLLOWS", "COLLECTED_BY",
                            "CHAIN_OF_CUSTODY", "SCENE_CONTAINS",
                        ],
                    },
                    "properties": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "role": {"type": "string"},
                            "weight": {"type": "string"},
                        },
                        "required": ["description"],
                        "additionalProperties": False,
                    },
                    "confidence": {"type": "number"},
                    "evidence_span": {"type": "string"},
                },
                "required": [
                    "source_entity_id", "target_entity_id",
                    "relationship_type", "properties", "confidence", "evidence_span",
                ],
                "additionalProperties": False,
            },
        },
    },
    "required": ["relationships"],
    "additionalProperties": False,
}

# --- Multi-modal Synthesis Schema (Text + Image combined) ---

MULTIMODAL_SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "unified_assessment": {"type": "string"},
        "cross_references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text_entity": {"type": "string"},
                    "image_finding": {"type": "string"},
                    "relationship": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["text_entity", "image_finding", "relationship", "confidence"],
                "additionalProperties": False,
            },
        },
        "additional_entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "entity_type": {"type": "string"},
                    "entity_id": {"type": "string"},
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["entity_type", "entity_id", "name", "description"],
                "additionalProperties": False,
            },
        },
        "additional_relationships": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "target_id": {"type": "string"},
                    "relationship_type": {"type": "string"},
                    "confidence": {"type": "number"},
                    "evidence": {"type": "string"},
                },
                "required": ["source_id", "target_id", "relationship_type", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
    },
    "required": [
        "unified_assessment", "cross_references", "additional_entities",
        "additional_relationships", "confidence", "reasoning",
    ],
    "additionalProperties": False,
}
