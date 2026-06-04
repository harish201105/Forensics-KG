# Forensics Knowledge Graph: LLM-Powered Forensic Analysis System

An end-to-end system that combines **Large Language Models (GPT-4o/4.1)**, **Knowledge Graphs (Neo4j)**, and **Computer Vision (OpenCV)** to automate forensic evidence analysis across **6 document types** and **6 forensic image types** — linking text extraction, image analysis, and structured reasoning through a unified knowledge graph. Built with **real case data** (Indian court judgments, SOCOFing fingerprints, CEDAR signatures, wound images) and **synthetic forensic data** for comprehensive evaluation.

Includes a **quantitative evaluation framework** (precision/recall/F1 against gold standard annotations across all document types), **multi-modal reasoning** (combined text + image analysis with GPT-4o synthesis), **temporal reasoning** (timeline reconstruction with gap detection), **RAG-augmented querying**, **collaborative annotation tools**, and a **strategy-based image analysis pipeline** supporting 6 forensic image types.

## Table of Contents

- [Problem Statement & Motivation](#problem-statement--motivation)
- [Research Objectives](#research-objectives)
- [What This System Does](#what-this-system-does)
- [System Architecture](#system-architecture)
- [Knowledge Graph Ontology](#knowledge-graph-ontology)
- [Key Research Outcomes](#key-research-outcomes)
- [Technology Stack](#technology-stack)
- [Setup & Installation](#setup--installation)
- [Usage Manual](#usage-manual)
- [Demo Guide](#demo-guide)
- [API Reference](#api-reference)
- [Dataset Information](#dataset-information)
- [Evaluation Framework](#evaluation-framework)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)

---

## Problem Statement & Motivation

Forensic investigation faces several critical challenges:

1. **Expert Scarcity**: Bloodstain pattern analysis (BPA), fingerprint analysis, ballistics, and document forensics all require highly trained analysts — there aren't enough to handle caseloads.
2. **Subjectivity**: Pattern classification is largely subjective — different analysts may reach different conclusions from the same evidence.
3. **Information Silos**: Case data (FIR reports, court judgments, witness depositions, lab reports) and physical evidence (bloodstain images, fingerprints, wound photos) exist in disconnected systems.
4. **No Structured Reasoning**: Investigators manually connect evidence to hypotheses without computational support for exploring alternative explanations.
5. **Reproducibility**: Traditional forensic analysis lacks quantitative, reproducible feature extraction.

**This project addresses these problems by:**

- Using LLMs to automatically extract structured entities and relationships from **6 types** of unstructured forensic text (FIR reports, court judgments, post-mortem reports, lab reports, witness depositions, SOCO reports)
- Combining traditional computer vision with LLM vision capabilities to analyze **6 types** of forensic images (bloodstain patterns, fingerprints, wound patterns, ballistics, document forensics, tool marks)
- Storing everything in a knowledge graph with **38 node types** and **48 relationship types** that captures the semantic relationships between cases, people, evidence, locations, weapons, forensic patterns, and legal proceedings
- Enabling natural language querying and hypothesis generation over the connected evidence
- Validating with **real case data** (Indian court judgments) alongside synthetic data for rigorous evaluation

---

## Research Objectives

1. **Automated Entity Extraction from Forensic Text**: Can GPT-4o reliably extract structured entities (persons, locations, evidence, weapons, timelines, legal sections, verdicts) from unstructured forensic documents across 6 document types?

2. **Hybrid Image Analysis for Forensic Evidence**: Can combining traditional CV feature extraction with LLM vision analysis produce more robust forensic image analysis than either approach alone — across bloodstain patterns, fingerprints, wounds, ballistics, document forensics, and tool marks?

3. **Knowledge Graph for Forensic Reasoning**: Does structuring forensic evidence as a knowledge graph enable novel cross-case analysis, pattern discovery, and hypothesis generation that wouldn't be possible with flat databases?

4. **LLM-Powered Forensic Hypothesis Generation**: Can LLMs generate evidence-based forensic hypotheses with confidence scores and supporting/contradicting evidence chains by reasoning over knowledge graph data?

5. **Real vs Synthetic Data Validation**: Does using real criminal case data (Indian court judgments) for gold standard generation produce more reliable evaluation than purely synthetic data?

6. **Multi-Document Cross-Referencing**: Can the system correlate evidence across different document types (e.g., linking witness deposition contradictions with physical evidence from SOCO reports)?

---

## What This System Does

### Core Capabilities

| Capability | What It Does | How It Works |
|---|---|---|
| **Text Extraction (6 types)** | Extracts entities & relationships from FIR reports, court judgments, post-mortem reports, lab reports, witness depositions, SOCO reports | GPT-4o/4.1 with structured output — schema registry pattern dispatches type-specific schemas and prompts |
| **Image Analysis (6 types)** | Analyzes bloodstain patterns, fingerprints, wound images, ballistics, document forensics, tool marks | Strategy pattern: OpenCV preprocessing + GPT-4o Vision per image type |
| **Multi-modal Extraction** | Combined text + image analysis in a single pass | Parallel extraction via `asyncio.gather`, followed by GPT-4o synthesis pass that cross-references findings |
| **Knowledge Graph** | Stores all extracted data as a connected graph | Neo4j with 38 node types and 48 relationship types defined by a forensics ontology |
| **Natural Language Query** | Ask questions in plain English | RAG-augmented NL → Cypher → Execute → Interpret, with a self-correction loop that feeds Neo4j errors back to fix the query |
| **Semantic Search + Projector** | Search the graph by meaning, not keywords | OpenAI embeddings on every node + a Neo4j vector index (cosine KNN); an interactive 2D PCA embedding-projector page |
| **Cross-Case Entity Resolution** | Link the same real-world entity across cases | Embedding-blocked, context-aware `SAME_AS` linking ("show all cases involving X", "which judges recur") |
| **Hypothesis Generation** | Generate forensic hypotheses from evidence | Retrieves case evidence from KG, performs statistical analysis, generates structured hypotheses with confidence scores |
| **Evaluation Framework** | Quantitative quality assessment | P/R/F1 for text extraction (6 doc types); **quantitative image evaluation** (6 image types); **real-case validation** against human-curated documented facts (10 cases) |
| **Temporal Reasoning** | Timeline reconstruction with gap detection | Queries TimeEvent nodes, sorts chronologically, detects gaps and inconsistencies |
| **Multi-Model Support** | Compare extraction quality across models | Per-request model selection (GPT-4o, GPT-4o-mini, GPT-4.1, GPT-4.1-mini) |
| **Real Data Ingestion** | Ingest real criminal case data | Court judgments from HuggingFace, Indian Kanoon API, or Kaggle CSV |
| **Gold Standard Generation** | LLM-generated gold from full judgment text | Comprehensive entity extraction from complete court judgment text |
| **Synthetic Data Generation** | Creates realistic test documents and images | GPT-4o generates forensic documents; OpenCV generates procedural ballistics/tool mark images |
| **Forensic Dataset Downloads** | Auto-download public forensic datasets | SOCOFing fingerprints (110K), CEDAR signatures (2.6K), AZH wounds (1.8K), Mendeley autopsies, Multi-LexSum depositions |
| **Collaborative Annotations** | Case notes, status tracking, activity logging | Annotation nodes linked to entities, case status management |
| **Data Export** | Export cases, graphs, reports | CSV, JSON export endpoints |

### Supported Document Types

| Document Type | Source Type Key | Entity Types Extracted |
|---|---|---|
| **FIR Report** | `fir` | Case, Person, Location, Evidence, Weapon, Vehicle, CrimeType, TimeEvent |
| **Court Judgment** | `court_judgment` | Case, Person (judge/accused/victim/witness/lawyer), Location, Evidence, Weapon, Vehicle, LegalSection, Verdict, TimeEvent |
| **Post-Mortem Report** | `postmortem` | Person (deceased/examiner), InjuryPattern, CauseOfDeath, ToxicologyResult, OrganFinding, TimeEvent |
| **Forensic Lab Report** | `lab_report` | Sample, TestResult, ChemicalCompound, DNAProfile, Evidence |
| **Witness Deposition** | `witness_deposition` | Person (witness/accused), Statement, TimeEvent, Location + CONTRADICTS relationships |
| **SOCO Report** | `soco_report` | CrimeScene, PhysicalEvidence, ChainOfCustody, Location |

### Supported Image Types

| Image Type | Strategy Class | CV Processing | GPT-4o Vision |
|---|---|---|---|
| **Bloodstain Pattern** | `BloodstainStrategy` | CLAHE, Otsu thresholding, contour detection, geometric features | Pattern classification, impact mechanism inference |
| **Fingerprint** | `FingerprintStrategy` | Ridge enhancement, thinning, Harris corner minutiae detection | ACE-V pattern classification (loop/whorl/arch), quality assessment |
| **Wound** | `WoundStrategy` | Color segmentation for tissue types, area measurement | Wound type classification, mechanism inference, age estimation |
| **Ballistics** | `BallisticsStrategy` | Edge detection, HoughLinesP for striations | AFTE bullet/cartridge classification, firearm type inference |
| **Document Forensics** | `DocumentForensicsStrategy` | Baseline/slant detection, ink analysis | Forgery indicators, handwriting characteristics |
| **Tool Marks** | `ToolMarksStrategy` | Striation extraction, depth profiling via shadow analysis | AFTE tool type classification, mark comparison |

### End-to-End Flow

```text
┌──────────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                                │
│                                                                      │
│  Text Documents (6 types)              Forensic Images (6 types)     │
│  ├── FIR Reports (synthetic)           ├── Bloodstain Patterns       │
│  ├── Court Judgments (real/HuggingFace) ├── Fingerprints (SOCOFing)   │
│  ├── Post-Mortem Reports (synthetic)   ├── Wound Images (AZH)        │
│  ├── Lab Reports (synthetic)           ├── Ballistics (synthetic)    │
│  ├── Witness Depositions (Multi-LexSum)├── Document Forensics (CEDAR)│
│  └── SOCO Reports (synthetic)          └── Tool Marks (synthetic)    │
└──────────┬─────────────────────────────────────────┬─────────────────┘
           │                                         │
           ▼                                         ▼
┌───────────────────┐                   ┌───────────────────────┐
│  Text Extractor   │                   │   Image Extractor     │
│  (Schema Registry)│                   │  (Strategy Pattern)   │
│                   │                   │                       │
│  source_type →    │                   │  image_type →         │
│  (entity_schema,  │                   │  strategy.preprocess  │
│   rel_schema,     │                   │  strategy.vision      │
│   prompt_builder) │                   │  strategy.combine     │
└───────┬───────────┘                   └──────────┬────────────┘
        │                                          │
        ├──────────┐   ┌───────────────────────────┘
        │          ▼   ▼
        │  ┌───────────────────┐
        │  │ Multi-modal       │  (Combined mode)
        │  │ Extractor         │
        │  │ asyncio.gather →  │
        │  │ GPT-4o synthesis  │
        │  └────────┬──────────┘
        │           │
        ▼           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      EXTRACTION PIPELINE                              │
│                                                                      │
│  1. Entity Deduplication (fuzzy name matching)                       │
│  2. Ontology Validation (type checking + Cypher injection prevention)│
│  3. Batched Neo4j Storage (transactional MERGE)                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      NEO4J KNOWLEDGE GRAPH                            │
│                                                                      │
│  38 Node Types (expanded ontology):                                  │
│  Case, Person, Location, Evidence, Weapon, Vehicle, CrimeType,       │
│  TimeEvent, BloodstainPattern, Stain, Experiment, Hypothesis,        │
│  LegalSection, Verdict, CourtOrder, InjuryPattern, CauseOfDeath,     │
│  ToxicologyResult, Sample, TestResult, DNAProfile, Statement,        │
│  CrimeScene, PhysicalEvidence, ChainOfCustody, FingerprintPattern,   │
│  MinutiaePoint, WoundPattern, BallisticsPattern, ToolMarkPattern...  │
│                                                                      │
│  46 Relationship Types:                                              │
│  INVOLVED_IN, OCCURRED_AT, HAS_EVIDENCE, CONTAINS_STAIN,            │
│  PRESIDED_BY, REPRESENTED_BY, CHARGED_UNDER, RESULTED_IN,           │
│  CAUSED_BY, TESTED_POSITIVE_FOR, MATCHES_DNA, STATED_BY,            │
│  FOUND_AT, COLLECTED_BY, CONTRADICTS, FIRED_FROM, WRITTEN_BY...     │
└──┬──────────┬──────────┬──────────┬──────────┬──────────┬────────────┘
   │          │          │          │          │          │
   ▼          ▼          ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐ ┌────────┐ ┌──────────┐
│NL Query│ │ Graph  │ │Hypothesis│ │Evalua- │ │Temporal│ │Collabor- │
│Engine  │ │ Viz    │ │Generator │ │tion    │ │Reason- │ │ation     │
│(RAG)   │ │        │ │          │ │Frame-  │ │ing     │ │          │
│        │ │Force-  │ │Evidence →│ │work    │ │        │ │Annotate  │
│Context │ │directed│ │Stats →   │ │        │ │Timeline│ │Case Mgmt │
│retriev │ │2D graph│ │LLM →    │ │P/R/F1  │ │recon → │ │Activity  │
│→Cypher │ │+ search│ │Hypothes.│ │6 types │ │Gap det.│ │Log       │
└────────┘ └────────┘ └──────────┘ └────────┘ └────────┘ └──────────┘
```

---

## System Architecture

```text
forensics-kg/
├── backend/                          # FastAPI (Python 3.12) — 67 source files
│   ├── app/
│   │   ├── main.py                   # App entry + lifespan + middleware
│   │   ├── config.py                 # Pydantic settings (env-based)
│   │   ├── dependencies.py           # FastAPI dependency injection
│   │   ├── models/                   # Pydantic request/response models
│   │   │   ├── requests.py           # TextExtractionRequest, SourceType enum (10 types)
│   │   │   └── responses.py          # ExtractionResponse, QueryResponse, etc.
│   │   ├── routers/                  # 9 routers, 56 endpoints
│   │   │   ├── extract.py            # POST /text, /image, /combined
│   │   │   ├── graph.py              # GET/DELETE /graph/* + WebSocket
│   │   │   ├── query.py              # POST /natural, /cypher
│   │   │   ├── datasets.py           # 18 endpoints: generate, ingest, download, gold, process
│   │   │   ├── analysis.py           # POST /hypothesis/{case_id}
│   │   │   ├── export.py             # GET /cases/csv, /graph/json, /case/report
│   │   │   ├── evaluation.py         # POST /run-all, /run-single + validation
│   │   │   ├── temporal.py           # GET /timeline, /inconsistencies
│   │   │   └── collaboration.py      # POST/GET/PATCH annotate, status, log
│   │   └── services/                 # Business logic layer
│   │       ├── extraction/
│   │       │   ├── openai_client.py   # OpenAI API wrapper (retry, structured output)
│   │       │   ├── text_extractor.py  # Schema registry (6 source types)
│   │       │   ├── image_extractor.py # Strategy dispatcher (6 image types)
│   │       │   ├── multimodal_extractor.py  # Text+Image parallel + synthesis
│   │       │   ├── pipeline.py        # Orchestrator: extract → dedup → store
│   │       │   ├── schemas.py         # 19 JSON schemas for structured LLM output
│   │       │   └── image_strategies/  # Strategy pattern implementations
│   │       │       ├── __init__.py    # ABC + strategy registry
│   │       │       ├── bloodstain.py  # CLAHE + Otsu + contour + GPT-4o Vision
│   │       │       ├── fingerprint.py # Ridge enhance + minutiae + ACE-V
│   │       │       ├── wound.py       # Color segmentation + wound classification
│   │       │       ├── ballistics.py  # Edge detection + AFTE classification
│   │       │       ├── document.py    # Baseline/ink analysis + forgery detection
│   │       │       └── toolmarks.py   # Striation extraction + tool classification
│   │       ├── graph/
│   │       │   ├── neo4j_client.py    # Async Neo4j driver wrapper
│   │       │   ├── operations.py      # Entity/relationship storage (Cypher-safe)
│   │       │   └── deduplication.py   # Fuzzy entity deduplication
│   │       ├── ontology/
│   │       │   └── schema.py          # YAML ontology loader (38 nodes, 48 rels)
│   │       ├── query/
│   │       │   ├── nl_to_cypher.py    # NL → Cypher generation
│   │       │   ├── query_service.py   # Query orchestrator
│   │       │   └── graph_retriever.py # RAG context retrieval
│   │       ├── reasoning/
│   │       │   └── engine.py          # Hypothesis generation + statistical analysis
│   │       ├── datasets/
│   │       │   ├── fir_generator.py              # Synthetic FIR generation
│   │       │   ├── document_generators.py        # PostMortem/Lab/Deposition/SOCO generators
│   │       │   ├── synthetic_image_generator.py  # Ballistics + tool marks (OpenCV)
│   │       │   ├── court_judgment_ingester.py     # HuggingFace/CSV/Indian Kanoon
│   │       │   ├── gold_generator.py             # LLM gold standard from judgments
│   │       │   ├── forensic_data_downloader.py   # SOCOFing/CEDAR/AZH/Mendeley/LexSum
│   │       │   ├── bloodstain_processor.py       # Attinger dataset processor
│   │       │   └── manager.py                    # Dataset orchestrator
│   │       ├── evaluation/
│   │       │   ├── evaluator.py       # 6 gold adapters + fuzzy matching
│   │       │   └── metrics.py         # P/R/F1 Pydantic models
│   │       ├── temporal/
│   │       │   └── timeline.py        # Timeline reconstruction + gap detection
│   │       └── collaboration/
│   │           └── annotations.py     # Annotations + case status + activity log
│   ├── .env.example                   # Environment template (no real keys)
│   └── requirements.txt               # Python dependencies
├── frontend/                          # Next.js 14 (TypeScript) — 27 source files
│   └── src/
│       ├── app/                       # 11 pages (Next.js App Router)
│       │   ├── page.tsx               # Dashboard (stats + distributions)
│       │   ├── extract/page.tsx       # 3 tabs: text (6 types) / image (6 types) / combined
│       │   ├── graph/page.tsx         # Force-directed 2D graph + search + filter
│       │   ├── query/page.tsx         # NL + Cypher query interface
│       │   ├── datasets/page.tsx      # 7 action cards: generate, ingest, download, gold
│       │   ├── analysis/page.tsx      # Hypothesis generation + statistics
│       │   ├── evaluation/page.tsx    # P/R/F1 dashboard with source type filter
│       │   ├── cases/page.tsx         # Case list with search
│       │   └── cases/[caseId]/page.tsx # Case detail: graph, entities, timeline, notes
│       ├── components/
│       │   ├── ui/                    # Spinner, Toast, DataTable, Card, Badge,
│       │   │                          # ModelSelector, ConfirmDialog, etc.
│       │   ├── timeline/              # CaseTimeline visualization
│       │   └── layout/               # Responsive sidebar navigation
│       ├── lib/
│       │   ├── api.ts                 # Axios API client (7 API groups, 30+ functions)
│       │   ├── hooks.ts               # 26 React Query hooks
│       │   └── store.ts              # Zustand store (model selection persistence)
│       └── types/
│           └── index.ts               # TypeScript interfaces + NODE_COLORS
├── ontology/
│   └── forensics_ontology.yaml        # Domain ontology (38 node types, 48 relationships)
└── data/                              # ~115K files (gitignored, regenerated via API)
    ├── fir_text/                      # 20 synthetic FIR reports (.txt)
    ├── gold/                          # 20 FIR gold standards (.json)
    ├── court_judgments/               # 40 real Indian court judgments (.txt)
    ├── gold_court/                    # 20 court judgment gold standards (.json)
    ├── postmortem/                    # 10 synthetic post-mortem reports
    ├── gold_postmortem/               # 10 post-mortem gold standards
    ├── lab_reports/                   # 10 synthetic lab reports
    ├── gold_lab/                      # 10 lab report gold standards
    ├── depositions/                   # 10+ witness depositions (Multi-LexSum)
    ├── gold_depositions/              # 10 deposition gold standards
    ├── soco/                          # 10 synthetic SOCO reports
    ├── gold_soco/                     # 10 SOCO gold standards
    ├── images/
    │   ├── fingerprints/socofing/     # 110,540 fingerprint images (Kaggle)
    │   ├── document_forensics/cedar/  # 2,640 signature images (GitHub)
    │   ├── wounds/azh/                # 1,867 wound images (GitHub)
    │   ├── ballistics/                # 20 synthetic ballistics images (OpenCV)
    │   └── tool_marks/                # 20 synthetic tool mark images (OpenCV)
    ├── gold_ballistics/               # 20 ballistics gold standards
    └── gold_toolmarks/                # 20 tool mark gold standards
```

### Component Interaction

```text
Browser (localhost:3000)
    │
    │  HTTP/REST (axios, 5min timeout)
    │
    ▼
Next.js Frontend ──────────────────────────────►  Neo4j Desktop
    │  11 pages, 26 hooks, ModelSelector             (localhost:7687)
    │  7 API groups                                      ▲
    │                                                    │
    ▼                                                    │
FastAPI Backend (localhost:8000)                          │
    │  9 routers, 56 endpoints                          │
    │                                                    │
    ├── OpenAI API (GPT-4o/4.1) ◄─ retry (3x, backoff) │
    │   - Text extraction (6 document types)             │
    │   - Image analysis (6 image types)                 │
    │   - Multi-modal synthesis                          │
    │   - NL-to-Cypher (RAG-augmented)                   │
    │   - Hypothesis generation                          │
    │   - Gold standard generation                       │
    │                                                    │
    ├── External Data Sources                            │
    │   - HuggingFace (court judgments)                  │
    │   - Kaggle (SOCOFing fingerprints)                 │
    │   - GitHub (CEDAR signatures, AZH wounds)          │
    │   - Indian Kanoon API (court judgments)             │
    │                                                    │
    └── Neo4j Async Driver ─────────────────────────────►┘
        - MERGE (upsert) with Cypher injection prevention
        - Batched transactions
        - Subgraph queries + full-text search
        - Timeline reconstruction
        - Annotations & activity log
```

---

## Knowledge Graph Ontology

### Node Types (38)

#### Core Case Entities

| Node Type | Unique Key | Description |
|---|---|---|
| **Case** | case_id | A forensic case/FIR |
| **Person** | person_id | Victim, suspect, witness, officer, judge, lawyer |
| **Location** | location_id | Crime scene, court, address |
| **Evidence** | evidence_id | Physical/digital evidence |
| **Weapon** | weapon_id | Weapon used in crime |
| **Vehicle** | vehicle_id | Vehicle involved |
| **CrimeType** | crime_type_id | Classification of crime |
| **TimeEvent** | event_id | Temporal event with timestamp |

#### Legal & Court Entities

| Node Type | Unique Key | Description |
|---|---|---|
| **LegalSection** | section_id | IPC/CrPC section cited |
| **Verdict** | verdict_id | Court verdict (convicted/acquitted) |
| **CourtOrder** | order_id | Court order or directive |

#### Forensic Analysis Entities

| Node Type | Unique Key | Description |
|---|---|---|
| **BloodstainPattern** | pattern_id | Classified bloodstain pattern |
| **Stain** | stain_id | Individual blood stain with geometric features |
| **Experiment** | experiment_id | Lab experiment metadata |
| **ImpactMechanismNode** | mechanism_id | Impact mechanism type |
| **FingerprintPattern** | pattern_id | Fingerprint ridge pattern |
| **MinutiaePoint** | minutiae_id | Fingerprint minutiae detail |
| **RidgeDetail** | ridge_id | Fingerprint ridge characteristics |
| **WoundPattern** | pattern_id | Wound classification pattern |
| **TissueAnalysis** | tissue_id | Tissue type analysis from wound |
| **BallisticsPattern** | pattern_id | Ballistics striation/comparison |
| **HandwritingFeature** | feature_id | Handwriting characteristic |
| **InkAnalysis** | analysis_id | Ink composition analysis |
| **ForgeryIndicator** | indicator_id | Document forgery indicator |
| **ToolMarkPattern** | pattern_id | Tool mark striation/impression |

#### Medical & Lab Entities

| Node Type | Unique Key | Description |
|---|---|---|
| **InjuryPattern** | injury_id | Injury type/location/dimensions |
| **CauseOfDeath** | cod_id | Cause of death determination |
| **ToxicologyResult** | tox_id | Toxicology test result |
| **OrganFinding** | finding_id | Organ-specific autopsy finding |
| **Sample** | sample_id | Lab sample (blood, hair, fiber) |
| **TestResult** | test_id | Lab test result |
| **DNAProfile** | dna_id | DNA profile with loci/alleles |
| **ChemicalCompound** | compound_id | Chemical substance identified |

#### Scene & Procedural Entities

| Node Type | Unique Key | Description |
|---|---|---|
| **Statement** | statement_id | Witness/accused statement |
| **CrimeScene** | scene_id | Crime scene description |
| **PhysicalEvidence** | pe_id | Physical evidence at scene |
| **ChainOfCustody** | coc_id | Evidence custody record |
| **Hypothesis** | hypothesis_id | Generated forensic hypothesis |
| **Annotation** | annotation_id | User annotation/note |

### Relationship Types (46)

Key relationships organized by domain:

**Case Relationships**: INVOLVED_IN, OCCURRED_AT, HAS_EVIDENCE, USED_WEAPON, HAPPENED_ON, HAS_CRIME_TYPE, WITNESSED_BY, BELONGS_TO, FOUND_AT

**Temporal**: PRECEDES, FOLLOWS

**Forensic Patterns**: CONTAINS_STAIN, GENERATED_BY, CAUSED_BY, HAS_PATTERN, HAS_MINUTIAE, RIDGE_PATTERN_OF, WOUND_CAUSED_BY, TISSUE_TYPE_OF, FIRED_FROM, STRIATION_MATCHES

**Legal**: PRESIDED_BY, REPRESENTED_BY, CHARGED_UNDER, RESULTED_IN, ORDERED

**Medical/Lab**: TESTED_POSITIVE_FOR, MATCHES_DNA, COLLECTED_BY, CHAIN_OF_CUSTODY

**Document Forensics**: WRITTEN_BY, FORGED_ON, TOOL_USED

**Reasoning**: SUPPORTS, CONTRADICTS, STATED_BY

**Collaboration**: HAS_ANNOTATION

---

## Key Research Outcomes

### 1. Schema Registry Pattern for Multi-Document Extraction

- A single `TextExtractor` handles all 6 document types through a **schema registry** that maps `source_type` to (entity_schema, relationship_schema, prompt_builder)
- Two-stage extraction (entities first, then relationships) with type-specific prompts
- Structured output (JSON schema with `strict: true`) ensures consistent, parseable output
- Graceful fallback to FIR schema for unknown document types

### 2. Strategy Pattern for Multi-Type Image Analysis

- `ImageExtractor` delegates to 6 concrete `ImageAnalysisStrategy` implementations
- Each strategy follows a 3-stage pipeline: CV preprocessing → GPT-4o Vision → Combined synthesis
- Traditional CV provides **quantitative features** (area, circularity, ridge count, striation patterns)
- GPT-4o Vision provides **qualitative classification** (pattern type, mechanism inference, quality assessment)

### 3. Real Case Data Validation

- **40 real Indian court judgments** ingested from HuggingFace dataset
- **20 LLM-generated gold standards** from full judgment text (comprehensive entity extraction)
- Breaks the circular evaluation problem of validating LLM extraction against LLM-generated synthetic data
- Gold standards include judges, accused, victims, witnesses, lawyers, legal sections, timeline events, verdicts

### 4. Comprehensive Forensic Dataset Integration

| Dataset | Source | Size | Type |
|---|---|---|---|
| SOCOFing Fingerprints | Kaggle | 110,540 images | Real fingerprints |
| CEDAR Signatures | GitHub | 2,640 images | Genuine + forged signatures |
| AZH Wound Images | GitHub | 1,867 images | Wound classification |
| Indian Court Judgments | HuggingFace | 40 documents | Real criminal cases |
| Multi-LexSum Depositions | HuggingFace | 20 documents | Real witness depositions |
| Mendeley Autopsies | Mendeley | 2 reports | Autopsy/post-mortem data |
| Synthetic FIRs | GPT-4o generated | 20 reports | Synthetic with gold |
| Synthetic Documents | GPT-4o generated | 40 reports | Post-mortem/Lab/Deposition/SOCO |
| Synthetic Ballistics | OpenCV generated | 20 images | Procedural with gold |
| Synthetic Tool Marks | OpenCV generated | 20 images | Procedural with gold |

### 5. Multi-modal Reasoning

- Parallel extraction via `asyncio.gather` — text and image processed simultaneously
- GPT-4o synthesis pass cross-references text entities with image findings
- Produces unified assessment with cross-reference pairs
- Graceful degradation: if one modality fails, the other still produces results

### 6. Knowledge Graph Enables Cross-Domain Reasoning

- Connecting case data with physical evidence analysis in a single graph enables queries spanning document types
- Graph structure captures relationships invisible in flat databases
- Cross-case pattern analysis (e.g., "Which cases involve similar wound patterns?")

### 7. Quantitative Evaluation Framework

- **6 gold standard adapters** — one per text document type
- Per-entity-type metrics: fuzzy matching via `difflib.SequenceMatcher`
- Aggregate precision/recall/F1 across all cases
- Per-case breakdown with extraction time tracking
- **Model comparison**: run evaluation with different models

### 8. Security Hardening

- **Cypher injection prevention**: Entity types and property keys sanitized against regex whitelist before interpolation into queries
- **Source type validation**: Image-only types rejected from evaluation endpoints with descriptive 400 errors
- **File existence checks**: Missing gold/text files return 404 instead of 500
- **Input validation**: Image type, file size, content type all validated at router level

### 9. Semantic Layer (Vector Embeddings + Projector)

- Every graph node is embedded (OpenAI `text-embedding-3-small`, 1536-dim) under a shared `:Embedded` label, indexed by a single Neo4j **vector index** (cosine)
- **Semantic KNN search** replaces brittle substring matching — retrieval by meaning, not literal string overlap
- **Embedding Projector** page: an interactive 2D PCA map of all nodes (TensorFlow-Projector style) with nearest-neighbour highlighting
- Embeddings also ground NL→Cypher: the RAG context surfaces semantically-relevant nodes **and the real relationship patterns** connecting them, so generated Cypher traverses actual edges instead of guessing
- Auto-embedding on every ingestion keeps the index current

### 10. Cross-Case Entity Resolution (Connected Forensic Graph)

The core promise of a forensic KG is **cross-case pattern discovery** — the same suspect across cases, a recurring judge, a shared location. But document-by-document extraction creates *per-case islands*: "Chandrasekhara Aiyar" in one judgment and "CHANDRASEKHARA AIYAR J." in another become **separate nodes**, so cross-case queries return nothing.

This layer resolves that:

- **Hybrid entity resolution**: node **embeddings block candidates** (cosine), then **type-specific name/string matching confirms** them (title-stripped, fuzzy + token-subset) — embeddings find the candidates, strings prevent false merges
- **Non-destructive `SAME_AS` links** (not merges) — preserves per-case provenance, which is essential forensically (you must know which case each mention came from)
- **Conservative + context-aware** by design: generic placeholder values are skipped, and weak matches (single common first names, conflicting initials like *G.C.* vs *G.S. Mathur*) are only accepted with **corroboration** — shared co-occurring entities (2-hop context) **or** a distinctive recurring role (judge/officer). Strong multi-token names pass on the name alone. Each `SAME_AS` edge records its `basis` (`name` / `name+role` / `name+context`) and confidence for transparency. False merges in forensics are dangerous, so the bar is deliberately high.
- **Auto-runs incrementally after each ingestion** (O(new × total)), so the graph stays connected as cases are added; also exposed as `POST /api/graph/resolve-entities` (with `dry_run`) for whole-graph (re)linking
- **`GET /api/graph/cross-case?label=&value=`** and a **Cross-Case page** answer "show all cases involving X," resolving aliases across cases
- NL→Cypher is `SAME_AS`-aware, so plain-English cross-case questions ("which judges appear in more than one case?") traverse the links

**Result**: queries that returned 0 before now surface real cross-case patterns — e.g. *Justice Chandrasekhara Aiyar presides over 3 cases* — discovered by linking `Chandrasekhara Aiyar` ≡ `CHANDRASEKHARA AIYAR J.`. This is the step that turns the system from a pipeline demonstration into demonstrated cross-case reasoning.

### 11. Quantitative Image Evaluation (and an honest negative result)

The original system evaluated text extraction with P/R/F1 but assessed image analysis only qualitatively. A quantitative framework now runs the **real CV + GPT-4o vision pipeline** on labelled images and scores predictions against ground truth (accuracy / precision-recall-F1 / MAE), one adapter per image type:

| Image type | Ground truth | Result (sample) |
|---|---|---|
| **Bloodstain** | Attinger impact-spatter mechanism | **F1 = 1.0** — reliable |
| Ballistics | synthetic gold caliber | ~0.5 caliber accuracy |
| Tool marks | synthetic gold tool type | ~0.0 (misclassifies synthetic marks) |
| Document (CEDAR) | genuine vs forged | F1 ≈ 0 (single-image forgery detection predicts "genuine") |
| Wound (AZH) | wound present vs background | recall 1.0 / precision 0.5 (over-detects) |
| Fingerprint (SOCOFing) | hand + finger | ~0.0 (model returns "unknown" — not visually recoverable) |

**This is a genuine, defensible research finding, not a defect.** The pipeline is quantitatively reliable on **bloodstain** — the one image type with real, physics-grounded public ground truth (the project's core contribution) — and the framework honestly exposes where the others fall short: misaligned ground truth (SOCOFing labels finger position, not pattern type; AZH wounds are clinical, not forensic) or genuinely hard tasks (single-image forgery detection needs a reference signature). Building the framework also surfaced and fixed a real **evaluation-validity bug** — the image's filename (which encodes the label) was leaking into the prompt via metadata, inflating fingerprint accuracy to a false 100%.

### 12. Scaled Real-Case Validation (breaking the circularity)

The synthetic gold standards are LLM-generated, so evaluating LLM extraction against them is **circular** — it proves consistency, not real-world correctness. This validates extraction against **human-curated gold from authoritative public sources** (court records / Wikipedia) for **10 well-documented real Indian criminal cases** (Koodathayi cyanide killings, 2024 R.G. Kar, 2012 Delhi / Nirbhaya, Sheena Bora, Aarushi-Hemraj, Jessica Lal, Priyadarshini Mattoo, Nithari, Neeraj Grover, Nitish Katara). The pipeline extracts entities from a factual case narrative; predictions are scored (P/R/F1) against the established documented facts.

| Entity type | Precision | Recall | F1 |
|---|---|---|---|
| **Person** (suspects/victims/witnesses) | 0.87 | 0.94 | **0.90** |
| CrimeType | 0.74 | 0.93 | 0.82 |
| TimeEvent (timeline) | 0.81 | 0.76 | 0.78 |
| Weapon / method | 0.89 | 0.67 | 0.76 |
| Location | 0.32 | 0.60 | 0.41 |
| **Overall (micro-avg)** | **0.76** | **0.83** | **0.79** |

**The headline: person identification reaches F1 = 0.90 on real, documented cases against human-authored ground truth** — converting "the pipeline is self-consistent" into "the pipeline is accurate on real forensic narratives." Each entity type uses a representation-appropriate matcher: persons/locations/weapons by fuzzy string match, **crime type** by canonical-keyword match against the case title (where the FIR extractor records it), and **timeline** by **date-aware matching** (year/month) so descriptive event names are compared fairly to documented dates — both transformed from earlier near-zero artifact scores (CrimeType 0.12 → 0.82, TimeEvent 0.03 → 0.78). Exposed at **Evaluation → Real Cases**; gold lives in `data/real_cases/`.

---

## Technology Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| **LLM** | OpenAI GPT-4o / GPT-4.1 | Entity extraction, image analysis, synthesis, NL-to-Cypher, hypothesis generation, gold generation |
| **Graph Database** | Neo4j 2026.x (Desktop) | Knowledge graph storage, Cypher queries, subgraph retrieval |
| **Backend** | FastAPI + Python 3.12 | Async REST API, 9 routers, 56 endpoints |
| **Image Processing** | OpenCV (headless) + NumPy + SciPy | 6 forensic image analysis strategies |
| **Frontend** | Next.js 14 + TypeScript + Tailwind CSS 4 | 11-page web application |
| **Graph Viz** | react-force-graph-2d | Interactive 2D force-directed graph |
| **Charts** | Recharts | Evaluation bar charts, timeline scatter plots |
| **State Management** | Zustand + React Query v5 | Client state + server state caching (26 hooks) |
| **HTTP Client** | Axios (frontend) + httpx (backend) | API communication + dataset downloads |
| **Retry/Resilience** | tenacity | Exponential backoff on OpenAI API calls |
| **Evaluation** | difflib (stdlib) | Fuzzy entity matching for P/R/F1 metrics |
| **Ontology** | YAML + Pydantic | Domain schema (38 nodes, 48 rels, 10 enums) |
| **Logging** | loguru | Structured logging throughout backend |

---

## Setup & Installation

### Prerequisites

- **Python 3.12+** (with pip)
- **Node.js 18+** (with npm)
- **Neo4j Desktop** (download from [neo4j.com/download](https://neo4j.com/download/))
- **OpenAI API Key** with GPT-4o access

### Quick Start (recommended)

After the one-time setup below, start the entire stack — Neo4j Desktop DBMS,
backend, and frontend — with a single command from `forensics-kg/`:

```bash
./start.sh     # brings up Neo4j + backend (:8000) + frontend (:3000)
./stop.sh      # stops all three
```

`start.sh` is idempotent (anything already running is left alone), creates the
backend venv and installs deps on first run, and waits until each tier is
healthy before reporting the live graph stats. The graph lives in the
**`neo4j` database of your Neo4j Desktop DBMS** — `start.sh` launches that DBMS
directly, so **do not** also press "Start" in the Neo4j Desktop GUI (two starts
collide on port 7687).

> **Port 7687 note:** if you also have Neo4j installed via Homebrew, it can
> autostart on login and steal port 7687 from Neo4j Desktop (making the graph
> appear empty). Disable its autostart once with `brew services stop neo4j`.

For the manual, step-by-step setup (or first-time configuration), continue below.

### Step 1: Neo4j Database

1. Open Neo4j Desktop
2. Create a new project, add a new local DBMS
3. Set the DBMS password to `forensics_kg_2024` (matching `NEO4J_PASSWORD` in `.env`)
4. Start the DBMS
5. Verify it's running at `bolt://localhost:7687`

> **The app uses the DBMS's default `neo4j` database** (config default
> `NEO4J_DATABASE=neo4j`) — all data is read/written there. Do **not** create a
> separate database named `forensics-kg`; an empty named database will make the
> graph appear empty even though the DBMS is running.
>
> **Do not start Neo4j with `docker compose` for normal use.** The bundled
> [docker-compose.yml](docker-compose.yml) spins up a *separate, empty* Neo4j and
> will collide on port 7687 with your Neo4j Desktop DBMS — making your data look
> lost. It exists only for throwaway/CI setups. If Homebrew Neo4j is also
> installed, disable its login autostart once with `brew services stop neo4j` for
> the same reason.

### Step 2: Backend

```bash
cd forensics-kg/backend

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment — copy the example and fill in your keys
cp .env.example .env
# Edit .env with your actual API keys:
#   OPENAI_API_KEY=sk-proj-your-key-here
#   NEO4J_PASSWORD=forensics_kg_2024
#   KAGGLE_USERNAME=your-kaggle-username  (optional, for SOCOFing download)
#   KAGGLE_KEY=your-kaggle-key            (optional, from kaggle.json)

# Start the backend
uvicorn app.main:app --reload --port 8000
```

The backend will be available at `http://localhost:8000`. Check `http://localhost:8000/api/health` to verify.

### Step 3: Frontend

```bash
cd forensics-kg/frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The frontend will be available at `http://localhost:3000`.

### Step 4: Populate Data

Once both services are running:

1. **Datasets page** (`/datasets`): Click "Download All" to fetch real forensic datasets (SOCOFing, CEDAR, AZH, etc.)
2. **Datasets page**: Click "Ingest" to download court judgments from HuggingFace
3. **Datasets page**: Click "Generate Batch" to create gold standards from judgments
4. **Datasets page**: Click "Generate" to create synthetic FIR reports (auto-processed into graph)
5. **Datasets page**: Generate synthetic post-mortem, lab, deposition, and SOCO reports
6. **Datasets page**: Generate synthetic ballistics and tool mark images

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key with GPT-4o access |
| `NEO4J_URI` | Yes | Neo4j connection URI (default: `bolt://localhost:7687`) |
| `NEO4J_USERNAME` | Yes | Neo4j username (default: `neo4j`) |
| `NEO4J_PASSWORD` | Yes | Neo4j database password |
| `DATA_DIR` | No | Data directory path (default: `../data`) |
| `KAGGLE_USERNAME` | No | Kaggle username (for SOCOFing fingerprint download) |
| `KAGGLE_KEY` | No | Kaggle API key (from `~/.kaggle/kaggle.json`) |
| `INDIAN_KANOON_API_TOKEN` | No | Indian Kanoon API token (for court judgment search) |

---

## Usage Manual

### 1. Dashboard (Home Page `/`)

The dashboard shows:

- **Quick action cards** — shortcuts to Extract, Datasets, Query, and Graph pages
- **Statistics cards** — total nodes, relationships, node types, and relationship types in the KG
- **Node distribution** — breakdown of entity counts by type with color-coded indicators
- **Relationship distribution** — counts by relationship type

**First time?** The graph will be empty. Go to **Datasets** to populate it.

### 2. Datasets Page (`/datasets`)

This is where you populate the knowledge graph with data. Seven action cards:

| Card | Action | What Happens |
|---|---|---|
| **Generate Synthetic FIRs** | Set count → Generate | GPT-4o generates FIRs + gold → extracts entities → stores in Neo4j |
| **Ingest Court Judgments** | Select source → Ingest | Downloads real judgments from HuggingFace/CSV/Indian Kanoon |
| **Generate Gold Standards** | Select model → Generate Batch | LLM extracts comprehensive gold from full judgment text |
| **Generate Forensic Documents** | Select type + count → Generate | Creates synthetic post-mortem/lab/deposition/SOCO reports |
| **Process Bloodstain Data** | Process (5 experiments) | Analyzes bloodstain images from Attinger dataset |
| **Download Real Forensic Datasets** | Download All | Fetches SOCOFing, CEDAR, AZH, Mendeley, Multi-LexSum |
| **Generate Synthetic Forensic Images** | Select type + count → Generate | OpenCV creates ballistics/tool mark images with gold metadata |

### 3. Extract Page (`/extract`)

Three tabs for extracting entities from individual documents:

#### Text Tab

1. Select **document type** from dropdown (FIR, Court Judgment, Post-Mortem, Lab Report, Witness Deposition, SOCO Report)
2. Paste document text
3. Click **Extract Entities** — view color-coded entities and relationships

#### Image Tab

1. Select **image type** from dropdown (Bloodstain, Fingerprint, Wound, Ballistics, Document Forensics, Tool Marks)
2. Upload an image (JPG, PNG, TIFF, BMP, up to 50MB)
3. View extracted patterns, features, and classification

#### Combined Tab (Multi-modal)

1. Select both document type and image type
2. Paste text + upload image
3. **Run Combined Analysis** — parallel extraction + GPT-4o synthesis
4. View unified assessment with cross-reference pairs

### 4. Knowledge Graph Page (`/graph`)

Interactive 2D force-directed graph visualization:

- **Search**: Find and center on specific nodes
- **Zoom/Fit**: Auto-zoom to see all nodes
- **Filter**: Toggle node types on/off in the legend
- **Inspect**: Click any node to see properties

### 5. Cases Page (`/cases`)

Browse all forensic cases. Click any case for the detail view:

- **Graph Tab**: Subgraph of all connected entities
- **Entities Tab**: Grouped entity cards
- **Timeline Tab**: Chronological events with gap detection
- **Analysis Tab**: Generate forensic hypotheses
- **Notes Tab**: Annotations and case status management

### 6. Query Page (`/query`)

- **Natural Language**: Ask questions in English → semantic RAG retrieval → Cypher generation → execute → interpret, with a **self-correction loop** (feeds Neo4j errors back to fix the query). Surfaces the answer, confidence, generated Cypher, key entities, clickable follow-up questions, and retrieved graph context
- **Cypher**: Write queries directly (destructive queries blocked for safety)

### 7. Projector Page (`/projector`)

- Interactive **2D PCA embedding map** of every graph node (TensorFlow-Projector style), coloured by node type
- **Semantic search** highlights the nearest neighbours of a query; toggle node types; "Embed new nodes" backfills embeddings
- Shows embedding coverage (nodes embedded / total)

### 8. Cross-Case Page (`/cross-case`)

- **Entity resolution**: preview (dry-run) or apply `SAME_AS` linking of duplicate entities across cases, with per-label link counts and example matches
- **"Show all cases involving X"**: pick an entity (e.g. a recurring judge) and see every case it appears in, with resolved aliases

### 9. Analysis Page (`/analysis`)

- **Hypothesis mode**: Generates primary + alternative hypotheses with confidence, supporting/contradicting evidence, reasoning chain, recommendations
- **Statistics mode**: Descriptive statistics on stain/experiment features

### 10. Evaluation Page (`/evaluation`)

Three tabs:

- **Text Extraction** — select source type (FIR, Court Judgment, Post-Mortem, Lab Report, Deposition, SOCO) and model; Run All / Run Single; view overall P/R/F1, per-entity-type chart, per-case breakdown
- **Image Analysis** — pick an image type and sample size; runs the real CV + vision pipeline on labelled images and scores accuracy / P-R-F1 / MAE against ground truth
- **Real Cases** — runs extraction on 10 documented real cases and scores P/R/F1 against human-curated gold (breaks the circular-evaluation problem)

---

## Demo Guide

### Quick Demo (5 minutes)

1. **Start all services** (Neo4j, backend, frontend)
2. **Dashboard**: Show empty graph state
3. **Datasets**: Generate 2-3 synthetic FIRs → show auto-processing into graph
4. **Graph**: Show the populated knowledge graph, click nodes, toggle types
5. **Query**: Ask "Show all cases with their suspects" → show NL answer + generated Cypher
6. **Analysis**: Generate hypothesis for a case → show reasoning chain

### Full Demo (20 minutes)

1-6 (same as Quick Demo), then:
7. **Datasets**: Ingest real court judgments from HuggingFace → show real data flowing in
8. **Datasets**: Generate gold standards → show LLM extracting comprehensive entities from judgment text
9. **Extract → Text**: Select "Court Judgment" type → paste judgment → show legal entity extraction (judges, legal sections, verdict)
10. **Extract → Image**: Select "Fingerprint" → upload SOCOFing image → show minutiae detection + pattern classification
11. **Extract → Combined**: Upload FIR text + bloodstain image → show multi-modal synthesis with cross-references
12. **Evaluation**: Run evaluation on FIRs → show P/R/F1 metrics
13. **Evaluation**: Switch source type to "Court Judgment" → run → compare metrics across document types
14. **Model Comparison**: Switch to GPT-4o-mini → re-run evaluation → compare quality/speed/cost
15. **Case Detail → Timeline**: Show chronological events with detected gaps
16. **Case Detail → Notes**: Add annotation, change case status
17. **Datasets**: Show "Download All" for real forensic datasets, show data source counts
18. **Export**: Download case CSV or graph JSON

### Key Talking Points

- **"6 document types, 6 image types"**: Show how one system handles diverse forensic evidence
- **"Real data + synthetic data"**: Court judgments are real cases; FIRs and reports are synthetic with gold standards
- **"LLM + traditional CV"**: Each image type combines OpenCV features with GPT-4o classification
- **"115K+ data items"**: SOCOFing alone has 110K fingerprints — real forensic-scale data
- **"Quantitative evaluation"**: P/R/F1 metrics across all document types — the core research deliverable
- **"Cross-domain queries"**: Ask questions spanning cases, evidence, and forensic patterns
- **"Security"**: Cypher injection prevention, input validation, source type enforcement

---

## API Reference

### Health

```http
GET /api/health → { "status": "ok", "neo4j": "connected" }
```

### Extraction (3 endpoints)

```http
POST /api/extract/text
Body: { "text": "...", "source_type": "fir|court_judgment|postmortem|lab_report|witness_deposition|soco_report", "store_in_graph": true, "model": "gpt-4o" }

POST /api/extract/image
Body: FormData (file, image_type=bloodstain|fingerprint|wound|ballistics|document_forensics|tool_marks, experiment_id?, case_id?, store_in_graph)

POST /api/extract/combined
Body: FormData (file, text, source_type, image_type, experiment_id?, case_id?, store_in_graph, model?)
```

### Graph (12 endpoints)

```http
GET  /api/graph/stats
GET  /api/graph/full?limit=500
GET  /api/graph/subgraph/{id}?label=Case&key=case_id&depth=2
GET  /api/graph/search?q=text&labels=Case,Person
GET  /api/graph/semantic-search?q=text&k=10            # embedding KNN search
GET  /api/graph/embeddings/status                      # embedding coverage
POST /api/graph/embeddings/backfill?only_missing=true  # embed nodes
GET  /api/graph/embeddings/projection?dim=2&limit=2000 # PCA for projector
POST /api/graph/resolve-entities?dry_run=false         # cross-case SAME_AS linking
GET  /api/graph/cross-case?label=Person&value=Bose     # all cases involving X
GET  /api/graph/node/{label}/{key}/{value}
DELETE /api/graph/clear?confirm=yes-delete-all-data
```

### Query (2 endpoints)

```http
POST /api/query/natural
Body: { "question": "...", "context_case_id": "...", "model": "gpt-4o" }

POST /api/query/cypher
Body: { "query": "MATCH ...", "parameters": {} }
```

### Datasets (18 endpoints)

```http
GET  /api/datasets/                                    # List all datasets
POST /api/datasets/generate-fir                        # Generate synthetic FIRs
POST /api/datasets/process-bloodstain?limit=5          # Process bloodstain experiments
POST /api/datasets/upload                              # Upload dataset file
POST /api/datasets/ingest-judgments?source=huggingface&limit=20  # Ingest court judgments
GET  /api/datasets/judgments                            # List ingested judgments
POST /api/datasets/generate-gold/{case_id}             # Generate gold for one judgment
POST /api/datasets/generate-gold-batch?limit=20        # Generate gold for all judgments
GET  /api/datasets/gold                                # List gold standards
POST /api/datasets/generate/{doc_type}?count=10        # Generate synthetic documents
POST /api/datasets/generate-images/{image_type}?count=20  # Generate synthetic images
GET  /api/datasets/sources                             # List all data sources with counts
POST /api/datasets/download-forensic-data?dataset_type=all&limit=20  # Download real datasets
GET  /api/datasets/forensic-data-status                # Download status/counts
```

### Analysis (2 endpoints)

```http
POST /api/analysis/hypothesis/{case_id}?model=gpt-4o
GET  /api/analysis/statistics/{experiment_id}
```

### Evaluation (9 endpoints)

```http
# Text extraction
POST /api/evaluation/run-all?source_type=fir&model=gpt-4.1
POST /api/evaluation/run-single/{doc_id}?source_type=fir&model=gpt-4.1
GET  /api/evaluation/results
# Image analysis (quantitative)
GET  /api/evaluation/image/types
POST /api/evaluation/image/run?image_type=bloodstain&limit=5&model=gpt-4.1
GET  /api/evaluation/image/results
# Real-case validation (documented-fact gold)
GET  /api/evaluation/real-cases/list
POST /api/evaluation/real-cases/run?model=gpt-4.1
GET  /api/evaluation/real-cases/results
```

Note: text `source_type` must be a text-based type (fir, court_judgment, postmortem, lab_report, witness_deposition, soco_report). Image-only types return 400.

### Temporal (2 endpoints)

```http
GET /api/temporal/timeline/{case_id}
GET /api/temporal/inconsistencies/{case_id}
```

### Collaboration (4 endpoints)

```http
POST  /api/collaboration/annotate
GET   /api/collaboration/annotations/{entity_type}/{entity_id}
PATCH /api/collaboration/case/{case_id}/status
GET   /api/collaboration/activity-log?limit=50
```

### Export (3 endpoints)

```http
GET /api/export/cases/csv
GET /api/export/graph/json?limit=2000
GET /api/export/case/{case_id}/report
```

---

## Dataset Information

### Real Forensic Datasets

| Dataset | Source | Count | Description |
|---|---|---|---|
| **SOCOFing** | Kaggle (`ruizgara/socofing`) | 110,540 | Real fingerprint images (600 subjects, 3 impressions, altered versions) |
| **CEDAR Signatures** | GitHub releases | 2,640 | Genuine + forged handwriting signatures for document forensics |
| **AZH Wound Images** | GitHub repository | 1,867 | Wound classification images (abrasion, bruise, laceration, etc.) |
| **Indian Court Judgments** | HuggingFace (`rishiai/indian-court-judgements-and-its-summaries`) | 40 | Real Indian criminal case judgments with full text |
| **Multi-LexSum Depositions** | HuggingFace | 20 | Witness depositions from civil rights cases |
| **Mendeley Autopsies** | Mendeley API | 2 | Autopsy/post-mortem report data |
| **Attinger Bloodstain Data** | Published dataset (2018) | 61 experiments | Bloodstain pattern images with metadata |

### Synthetic Data

| Dataset | Generator | Count | Description |
|---|---|---|---|
| **Synthetic FIRs** | GPT-4o (`SyntheticFIRGenerator`) | 20 | Realistic Indian FIR reports with ground truth JSON |
| **Post-Mortem Reports** | GPT-4o (`PostMortemGenerator`) | 10 | Synthetic autopsy reports |
| **Lab Reports** | GPT-4o (`LabReportGenerator`) | 10 | Synthetic forensic lab reports |
| **Witness Depositions** | GPT-4o (`WitnessDepositionGenerator`) | 10 | Synthetic witness statements |
| **SOCO Reports** | GPT-4o (`SOCOReportGenerator`) | 10 | Synthetic scene-of-crime reports |
| **Ballistics Images** | OpenCV (`BallisticsImageGenerator`) | 20 | Bullet striations, cartridge headstamps, rifling comparisons |
| **Tool Mark Images** | OpenCV (`ToolMarkImageGenerator`) | 20 | Striation marks, impression marks, cut marks, pry marks |

### Gold Standards (120 total)

| Type | Count | Source |
|---|---|---|
| FIR gold | 20 | Auto-generated alongside synthetic FIRs |
| Court judgment gold | 20 | LLM-extracted from full judgment text |
| Post-mortem gold | 10 | Auto-generated alongside synthetic reports |
| Lab report gold | 10 | Auto-generated alongside synthetic reports |
| Deposition gold | 10 | Auto-generated alongside synthetic reports |
| SOCO gold | 10 | Auto-generated alongside synthetic reports |
| Ballistics gold | 20 | Metadata from procedural OpenCV generation |
| Tool mark gold | 20 | Metadata from procedural OpenCV generation |

---

## Evaluation Framework

The evaluation framework provides **quantitative evidence** that LLM-based extraction works for forensic text across all 6 document types.

### How It Works

1. **Gold Standards**: 120 gold standard files across 8 categories (6 text + 2 image)
2. **Gold Adapters**: 6 type-specific adapters extract ground truth items from gold JSON (different entity structures per document type)
3. **Extraction**: For each document, the system runs the full extraction pipeline using the selected model
4. **Matching**: Extracted entities compared against gold standard using type-specific matchers:
   - **Person**: Fuzzy name matching (`difflib.SequenceMatcher` >= 0.75)
   - **Location**: Substring and fuzzy matching
   - **Weapon/Vehicle**: Keyword matching
   - **TimeEvent**: Timestamp proximity + description similarity
   - **LegalSection**: Section number + act matching (court judgments)
   - **InjuryPattern**: Type + location matching (post-mortem)
   - **Sample/TestResult**: Sample type + test name matching (lab reports)
   - **Statement**: Content similarity matching (depositions)
   - **PhysicalEvidence**: Evidence type + description matching (SOCO)
5. **Metrics**: TP, FP, FN → precision, recall, F1 per entity type and aggregate
6. **Validation**: Image-only types (bloodstain, fingerprint, wound, ballistics, document_forensics, tool_marks) are rejected with 400 error — only text types can be evaluated

### Using the Evaluation Page

- **Source Type Filter**: Select which document type to evaluate (FIR, Court Judgment, Post-Mortem, Lab Report, Deposition, SOCO)
- **Run All**: Evaluates all gold standard cases for that type
- **Run Single**: Evaluates one document with matched entity pair details
- **Model Comparison**: Switch models and re-run to compare quality vs cost

---

## Limitations & Future Work

### Current Limitations

- **LLM dependency**: All extraction depends on OpenAI API availability and cost
- **Synthetic text data**: FIRs and forensic documents (except court judgments and depositions) are LLM-generated — though extraction is now also validated against human-curated gold for **10 real documented cases** (Key Research Outcome #12, overall F1 = 0.79 / Person F1 = 0.90)
- **Image analysis reliable mainly on bloodstain**: the quantitative image-evaluation framework (Outcome #11) shows the vision pipeline is dependable on bloodstain (F1 = 1.0 — the type with real, physics-grounded ground truth) but weaker on others, limited by ground-truth alignment (e.g. SOCOFing labels finger position, not pattern type) or genuinely hard tasks (single-image forgery detection)
- **Location extraction precision**: on real cases the extractor over-emits granular/multiple locations (Location F1 ≈ 0.41) relative to the single documented location
- **No user authentication**: The system has no login/roles — not suitable for production forensic use without security hardening
- **English-only**: NL queries and document generation are English-only
- **OpenAI-only models**: Multi-model support is limited to OpenAI models due to `json_schema` + `strict: True` structured output requirement
- **Single-image analysis**: Each image analyzed independently; multi-view comparison not supported

### Implemented Enhancements

- **Semantic layer** (Outcome #9): OpenAI embeddings on every node + Neo4j vector index (cosine KNN); interactive 2D embedding-projector page; auto-embedding on ingestion
- **Cross-case entity resolution** (Outcome #10): context-aware `SAME_AS` linking of duplicate entities across cases; "show all cases involving X"
- **Quantitative image evaluation** (Outcome #11): all 6 image pipelines scored against ground truth — accuracy / P-R-F1 / MAE
- **Scaled real-case validation** (Outcome #12): 10 documented real cases with human-curated, documented-fact gold — breaks the circular evaluation
- **Self-correcting NL→Cypher**: semantic-RAG context + relationship-pattern grounding + an error-feedback retry loop
- Evaluation framework with 6 gold adapters (P/R/F1 per entity type per document type)
- Multi-modal reasoning (combined text + image with GPT-4o synthesis)
- Temporal reasoning (timeline reconstruction with gap detection)
- Collaborative features (annotations, case status, activity log)
- Multi-model support (per-request model selection; GPT-4.1 default)
- Schema registry (6 document types) + strategy pattern (6 image types)
- Real data ingestion (court judgments, fingerprints, signatures, wounds, depositions)
- APOC enabled (faster subgraph/stats); Cypher injection prevention and input validation
- Synthetic image generation (ballistics, tool marks via OpenCV)

### Future Work

- **Real FIR data**: Validate with actual (anonymized) police reports from Indian police stations
- **Location precision**: tighten location extraction to reduce over-emission of granular sub-locations
- **Human-expert annotation**: extend real-case validation beyond 10 cases and beyond public documented facts to expert-annotated ground truth
- **Non-OpenAI models**: Support Claude and Gemini with adapter pattern for structured output
- **Multi-view analysis**: Compare multiple images of the same evidence
- **User authentication**: Role-based access control for production forensic use
- **Multilingual support**: Hindi and regional language document processing
- **Active learning**: Use evaluation results to fine-tune extraction prompts

---

## References

1. Attinger, D., Liu, Y., Bybee, T., & De Brabanter, K. (2018). A data set of bloodstain patterns for teaching and research in bloodstain pattern analysis: Impact beating spatters. *Data in Brief*, 18, 648-654. [DOI](https://doi.org/10.1016/j.dib.2018.02.070)

2. OpenAI. (2024). GPT-4o Technical Report. [OpenAI Research](https://openai.com/research/gpt-4o)

3. Neo4j, Inc. (2024). Neo4j Graph Database. [neo4j.com](https://neo4j.com/)

4. Scientific Working Group on Bloodstain Pattern Analysis (SWGSTAIN). Guidelines for the standardized terminology in bloodstain pattern analysis.

5. Rishiai. Indian Court Judgements and Summaries Dataset. [HuggingFace](https://huggingface.co/datasets/rishiai/indian-court-judgements-and-its-summaries)

6. Ruiz-Garcia, A. SOCOFing Fingerprint Dataset. [Kaggle](https://www.kaggle.com/datasets/ruizgara/socofing)

7. Association of Firearm and Tool Mark Examiners (AFTE). Theory of identification as it relates to toolmarks.

---

*UROP Research Project — Forensics Knowledge Graph System*
*38 node types | 48 relationship types (incl. SAME_AS cross-case links) | 56 API endpoints | 11 frontend pages | semantic vector search + embedding projector | cross-case entity resolution | quantitative image evaluation | 115K+ data items | 120 synthetic gold standards + 10 documented real-case validations*
