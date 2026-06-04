import re
from pathlib import Path, PurePosixPath
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from typing import List, Optional

from app.config import get_settings
from app.dependencies import get_neo4j_client, get_ontology_schema
from app.services.graph.neo4j_client import Neo4jClient
from app.services.graph.operations import GraphOperations
from app.services.ontology.schema import ForensicsOntologySchema
from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.text_extractor import TextExtractor
from app.services.extraction.image_extractor import ImageExtractor
from app.services.extraction.pipeline import ExtractionPipeline
from app.services.graph.deduplication import EntityDeduplicator
from app.services.graph.embedding_service import EmbeddingService
from app.services.graph.entity_resolution import EntityResolutionService
from app.services.datasets.fir_generator import SyntheticFIRGenerator
from app.services.datasets.bloodstain_processor import BloodstainDataProcessor
from app.services.datasets.manager import DatasetManager
from app.services.datasets.forensic_image_processor import ForensicImageProcessor
from app.services.datasets.court_judgment_ingester import CourtJudgmentIngester
from app.services.datasets.gold_generator import GoldStandardGenerator
from app.services.datasets.forensic_data_downloader import ForensicDataDownloader
from app.services.datasets.document_generators import (
    PostMortemGenerator,
    LabReportGenerator,
    WitnessDepositionGenerator,
    SOCOReportGenerator,
)
from app.services.datasets.synthetic_image_generator import (
    BallisticsImageGenerator,
    ToolMarkImageGenerator,
)
from app.models.requests import SyntheticFIRRequest
from app.models.responses import DatasetInfoResponse

router = APIRouter()


def _build_dataset_manager(
    client: Neo4jClient, schema: ForensicsOntologySchema
) -> DatasetManager:
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    fir_gen = SyntheticFIRGenerator(openai_client, data_dir=settings.data_dir)
    bloodstain_proc = BloodstainDataProcessor(settings.resolved_bloodstain_dirs)
    text_ext = TextExtractor(openai_client, schema)
    image_ext = ImageExtractor(openai_client, settings)
    graph_ops = GraphOperations(client, schema)
    dedup = EntityDeduplicator(client, schema)
    embedding_service = EmbeddingService(client, openai_client, settings.embedding_dimensions)
    resolution_service = EntityResolutionService(client)
    pipeline = ExtractionPipeline(
        text_ext, image_ext, graph_ops, dedup,
        openai_client=openai_client, embedding_service=embedding_service,
        resolution_service=resolution_service,
    )
    return DatasetManager(
        data_dir=settings.data_dir,
        fir_generator=fir_gen,
        bloodstain_processor=bloodstain_proc,
        pipeline=pipeline,
    )


@router.get("/", response_model=List[DatasetInfoResponse])
async def list_datasets(
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    manager = _build_dataset_manager(client, schema)
    datasets = manager.list_datasets()
    return [DatasetInfoResponse(**d) for d in datasets]


@router.post("/generate-fir")
async def generate_synthetic_fir(
    request: SyntheticFIRRequest,
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    manager = _build_dataset_manager(client, schema)
    result = await manager.generate_and_process_firs(
        count=request.count,
        crime_types=request.crime_types,
        include_bloodstain=request.include_bloodstain,
        neo4j_client=client,
    )
    return result


@router.post("/process-bloodstain")
async def process_bloodstain_data(
    limit: int = 5,
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    manager = _build_dataset_manager(client, schema)
    result = await manager.process_bloodstain_data(limit=limit)
    return result


@router.post("/process-images")
async def process_forensic_images(
    image_type: str = Query(
        ...,
        pattern="^(fingerprint|wound|document_forensics|ballistics|tool_marks)$",
    ),
    limit: int = Query(10, ge=1, le=200),
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    """Process downloaded/generated forensic images into the knowledge graph."""
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    text_ext = TextExtractor(openai_client, schema)
    image_ext = ImageExtractor(openai_client, settings)
    graph_ops = GraphOperations(client, schema)
    dedup = EntityDeduplicator(client, schema)
    embedding_service = EmbeddingService(client, openai_client, settings.embedding_dimensions)
    resolution_service = EntityResolutionService(client)
    pipeline = ExtractionPipeline(
        text_ext, image_ext, graph_ops, dedup,
        openai_client=openai_client, embedding_service=embedding_service,
        resolution_service=resolution_service,
    )

    processor = ForensicImageProcessor(settings.data_dir)
    return await processor.process_all(
        extraction_pipeline=pipeline,
        image_type=image_type,
        limit=limit,
    )


@router.get("/image-counts")
async def get_image_counts():
    """Return counts of available forensic images by type."""
    settings = get_settings()
    processor = ForensicImageProcessor(settings.data_dir)
    counts = {}
    for img_type in ["fingerprint", "wound", "document_forensics", "ballistics", "tool_marks"]:
        try:
            counts[img_type] = len(processor.discover_images(img_type))
        except Exception:
            counts[img_type] = 0
    return counts


@router.post("/process-judgments")
async def process_court_judgments(
    limit: int = Query(20, ge=1, le=100),
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    """Process ingested court judgment texts through extraction into the graph."""
    manager = _build_dataset_manager(client, schema)
    return await manager.process_court_judgments(limit=limit)


@router.post("/process-documents")
async def process_documents(
    doc_type: str = Query(
        ...,
        pattern="^(fir|postmortem|lab_report|witness_deposition|soco_report|court_judgment)$",
    ),
    limit: int = Query(20, ge=1, le=100),
    client: Neo4jClient = Depends(get_neo4j_client),
    schema: ForensicsOntologySchema = Depends(get_ontology_schema),
):
    """Process text documents of any type through extraction into the graph."""
    manager = _build_dataset_manager(client, schema)
    return await manager.process_documents(doc_type=doc_type, limit=limit)


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_type: str = "fir",
):
    settings = get_settings()

    # Sanitize filename: strip path components, remove dangerous characters
    raw_name = file.filename or "upload"
    safe_name = PurePosixPath(raw_name).name  # strip directory components
    safe_name = re.sub(r"[^\w\.\-]", "_", safe_name)  # keep only safe chars
    if not safe_name or safe_name.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")

    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    save_dir = settings.data_dir / "fir_text" if dataset_type == "fir" else settings.data_dir / "uploaded"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / safe_name
    save_path.write_bytes(content)
    return {"message": f"File saved to {save_path}", "filename": safe_name}


# --- Court Judgment Ingestion ---

@router.post("/ingest-judgments")
async def ingest_court_judgments(
    source: str = Query("huggingface", pattern="^(huggingface|csv|indian_kanoon)$"),
    csv_filename: Optional[str] = Query(None, description="CSV filename in data/ directory"),
    query: str = Query("murder FIR IPC", description="Search query for Indian Kanoon"),
    limit: int = Query(20, ge=1, le=200),
):
    """Ingest real criminal case judgments.

    Sources:
    - huggingface: Auto-downloads from HuggingFace (recommended, no auth needed)
    - csv: Reads from a local CSV file in data/ directory
    - indian_kanoon: Searches Indian Kanoon API (requires INDIAN_KANOON_API_TOKEN)
    """
    settings = get_settings()
    ingester = CourtJudgmentIngester(
        settings.data_dir,
        indian_kanoon_token=settings.indian_kanoon_api_token,
    )

    if source == "huggingface":
        judgments = await ingester.ingest_from_huggingface(limit=limit)
    elif source == "csv":
        csv_path = settings.data_dir / (csv_filename or "indian_court_judgments.csv")
        judgments = await ingester.ingest_from_csv(csv_path, limit=limit)
    else:
        judgments = await ingester.ingest_from_indian_kanoon(query=query, limit=limit)

    return {
        "message": f"Ingested {len(judgments)} criminal judgments from {source}",
        "source": source,
        "count": len(judgments),
        "case_ids": [j["case_id"] for j in judgments],
    }


@router.get("/judgments")
async def list_judgments():
    """List all ingested court judgments."""
    settings = get_settings()
    ingester = CourtJudgmentIngester(settings.data_dir, indian_kanoon_token="")
    return ingester.list_ingested()


@router.post("/generate-gold/{case_id}")
async def generate_gold_standard(
    case_id: str,
    model: Optional[str] = Query(None),
):
    """Generate gold standard from a court judgment's full text."""
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    generator = GoldStandardGenerator(openai_client, settings.data_dir)

    judgment_path = settings.data_dir / "court_judgments" / f"{case_id}.txt"
    if not judgment_path.exists():
        raise HTTPException(status_code=404, detail=f"Judgment {case_id} not found")

    full_text = judgment_path.read_text(encoding="utf-8")
    gold = await generator.generate_gold(case_id, full_text, model_override=model)

    return {
        "message": f"Generated gold standard for {case_id}",
        "case_id": case_id,
        "gold": gold,
    }


@router.post("/generate-gold-batch")
async def generate_gold_batch(
    limit: int = Query(20, ge=1, le=100),
    model: Optional[str] = Query(None),
):
    """Generate gold standards for all ingested judgments."""
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    generator = GoldStandardGenerator(openai_client, settings.data_dir)

    results = await generator.generate_batch(limit=limit, model_override=model)
    return {
        "message": f"Generated {len(results)} gold standards",
        "count": len(results),
        "case_ids": [r["case_id"] for r in results],
    }


@router.get("/gold")
async def list_gold_standards():
    """List all generated gold standard files."""
    settings = get_settings()
    openai_client = OpenAIClient(settings)
    generator = GoldStandardGenerator(openai_client, settings.data_dir)
    return generator.list_gold()


# --- Synthetic Document Generation ---

@router.post("/generate/{doc_type}")
async def generate_synthetic_documents(
    doc_type: str,
    count: int = Query(10, ge=1, le=50),
):
    """Generate synthetic forensic documents of a given type.

    Supported types: postmortem, lab_report, witness_deposition, soco_report
    """
    settings = get_settings()
    openai_client = OpenAIClient(settings)

    generators = {
        "postmortem": (PostMortemGenerator, "postmortem"),
        "lab_report": (LabReportGenerator, "lab_reports"),
        "witness_deposition": (WitnessDepositionGenerator, "depositions"),
        "soco_report": (SOCOReportGenerator, "soco"),
    }

    if doc_type not in generators:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown document type '{doc_type}'. Supported: {list(generators.keys())}",
        )

    gen_class, dir_name = generators[doc_type]
    generator = gen_class(openai_client)
    reports = await generator.generate_batch(count=count)
    await generator.save_batch(reports, settings.data_dir / dir_name)

    return {
        "message": f"Generated {len(reports)} {doc_type} documents",
        "doc_type": doc_type,
        "count": len(reports),
    }


@router.post("/generate-images/{image_type}")
async def generate_synthetic_images(
    image_type: str,
    count: int = Query(20, ge=1, le=100),
):
    """Generate synthetic forensic images of a given type.

    Supported types: ballistics, tool_marks
    """
    settings = get_settings()

    generators = {
        "ballistics": (
            BallisticsImageGenerator,
            settings.data_dir / "images" / "ballistics",
        ),
        "tool_marks": (
            ToolMarkImageGenerator,
            settings.data_dir / "images" / "tool_marks",
        ),
    }

    if image_type not in generators:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown image type '{image_type}'. Supported: {list(generators.keys())}",
        )

    gen_class, output_dir = generators[image_type]
    generator = gen_class(output_dir, data_dir=settings.data_dir)
    results = generator.generate_batch(count=count)

    return {
        "message": f"Generated {len(results)} {image_type} images",
        "image_type": image_type,
        "count": len(results),
    }


@router.get("/sources")
async def list_data_sources():
    """List all available data sources and their status."""
    settings = get_settings()
    sources = []

    # Text document sources
    dir_configs = [
        ("fir", "fir_text", "gold", "Synthetic FIR Reports"),
        ("court_judgment", "court_judgments", "gold_court", "Court Judgments"),
        ("postmortem", "postmortem", "gold_postmortem", "Post-Mortem Reports"),
        ("lab_report", "lab_reports", "gold_lab", "Forensic Lab Reports"),
        ("witness_deposition", "depositions", "gold_depositions", "Witness Depositions"),
        ("soco_report", "soco", "gold_soco", "SOCO Reports"),
    ]

    for source_type, data_dir, gold_dir, label in dir_configs:
        data_path = settings.data_dir / data_dir
        gold_path = settings.data_dir / gold_dir
        data_count = len(list(data_path.glob("*.txt"))) if data_path.exists() else 0
        gold_count = len(list(gold_path.glob("*.json"))) if gold_path.exists() else 0
        sources.append({
            "source_type": source_type,
            "label": label,
            "document_count": data_count,
            "gold_count": gold_count,
        })

    # Image sources
    image_configs = [
        ("fingerprints", "images/fingerprints/socofing", "gold_fingerprints", "SOCOFing Fingerprints"),
        ("signatures", "images/document_forensics/cedar", "gold_signatures", "CEDAR Signatures"),
        ("wounds", "images/wounds/azh", "gold_wounds", "AZH Wound Images"),
        ("ballistics", "images/ballistics", "gold_ballistics", "Ballistics Images"),
        ("tool_marks", "images/tool_marks", "gold_toolmarks", "Tool Mark Images"),
    ]

    for source_type, img_dir, gold_dir_name, label in image_configs:
        img_path = settings.data_dir / img_dir
        img_count = 0
        if img_path.exists():
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.BMP"):
                img_count += len(list(img_path.rglob(ext)))
        gold_path = settings.data_dir / gold_dir_name
        gold_count = len(list(gold_path.glob("*.json"))) if gold_path.exists() else 0
        sources.append({
            "source_type": source_type,
            "label": label,
            "document_count": img_count,
            "gold_count": gold_count,
        })

    return sources


# --- Forensic Data Downloads (public datasets, no auth) ---

@router.post("/download-forensic-data")
async def download_forensic_data(
    dataset_type: str = Query(
        "all",
        pattern="^(all|signatures|wounds|fingerprints|autopsies|depositions)$",
    ),
    limit: int = Query(20, ge=1, le=200),
):
    """Download real forensic datasets from public and Kaggle sources.

    Supported dataset_type values:
    - all: Download all available datasets
    - signatures: CEDAR signature verification images (genuine + forged)
    - wounds: AZH wound classification images
    - fingerprints: SOCOFing fingerprint images (requires Kaggle credentials)
    - autopsies: Mendeley autopsy / post-mortem report data
    - depositions: Multi-LexSum witness depositions from civil rights cases
    """
    settings = get_settings()
    downloader = ForensicDataDownloader(
        data_dir=settings.data_dir,
        kaggle_username=settings.kaggle_username,
        kaggle_key=settings.kaggle_key,
    )

    if dataset_type == "all":
        results = await downloader.download_all(limit_per_type=limit)
        total = sum(
            r.get("count", 0) for r in results.values() if r.get("status") == "ok"
        )
        return {
            "message": f"Forensic data download complete ({total} items across datasets)",
            "datasets": results,
            "total_items": total,
        }

    method_map = {
        "signatures": ("cedar_signatures", downloader.download_cedar_signatures, False),
        "wounds": ("azh_wounds", downloader.download_azh_wounds, False),
        "fingerprints": ("socofing_fingerprints", downloader.download_socofing_fingerprints, False),
        "autopsies": ("mendeley_autopsies", downloader.download_mendeley_autopsies, True),
        "depositions": ("multilexsum_depositions", downloader.download_multilexsum_depositions, True),
    }

    name, method, accepts_limit = method_map[dataset_type]

    if accepts_limit:
        count = await method(limit=limit)
    else:
        count = await method()

    return {
        "message": f"Downloaded {count} items for {name}",
        "dataset": name,
        "count": count,
        "status": downloader.status(),
    }


@router.get("/forensic-data-status")
async def forensic_data_status():
    """Return download counts for all forensic datasets."""
    settings = get_settings()
    downloader = ForensicDataDownloader(
        data_dir=settings.data_dir,
        kaggle_username=settings.kaggle_username,
        kaggle_key=settings.kaggle_key,
    )
    return downloader.status()
