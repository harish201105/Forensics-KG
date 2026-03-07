from pathlib import Path
from typing import List, Dict, Any, Optional
from loguru import logger
from app.services.datasets.fir_generator import SyntheticFIRGenerator
from app.services.datasets.bloodstain_processor import BloodstainDataProcessor
from app.services.datasets.forensic_image_processor import (
    ForensicImageProcessor,
    IMAGE_TYPE_CONFIGS,
)
from app.services.extraction.pipeline import ExtractionPipeline


class DatasetManager:
    """Manages all datasets: discovery, generation, processing."""

    def __init__(
        self,
        data_dir: Path,
        fir_generator: SyntheticFIRGenerator,
        bloodstain_processor: BloodstainDataProcessor,
        pipeline: ExtractionPipeline,
    ):
        self._data_dir = data_dir
        self._fir_gen = fir_generator
        self._bloodstain = bloodstain_processor
        self._pipeline = pipeline

    def list_datasets(self) -> List[Dict[str, Any]]:
        datasets = []

        # Bloodstain experiments
        experiments = self._bloodstain.discover_experiments()
        datasets.append({
            "dataset_id": "bloodstain_experiments",
            "name": "Bloodstain Pattern Experiments (2018 Dataset)",
            "type": "bloodstain",
            "record_count": len(experiments),
            "loaded": False,
            "metadata": {
                "categories": list(
                    set(
                        self._bloodstain._determine_category(str(e["experiment_id"]))
                        for e in experiments
                    )
                )
            },
        })

        # Synthetic FIR data
        fir_dir = self._data_dir / "fir_text"
        if fir_dir.exists():
            fir_files = list(fir_dir.glob("*.txt"))
            datasets.append({
                "dataset_id": "synthetic_fir",
                "name": "Synthetic FIR Reports",
                "type": "fir",
                "record_count": len(fir_files),
                "loaded": False,
                "metadata": {},
            })

        # Gold standard
        gold_dir = self._data_dir / "gold"
        if gold_dir.exists():
            gold_files = list(gold_dir.glob("*.json"))
            if gold_files:
                datasets.append({
                    "dataset_id": "gold_standard",
                    "name": "Gold Standard (Ground Truth)",
                    "type": "gold",
                    "record_count": len(gold_files),
                    "loaded": False,
                    "metadata": {},
                })

        # Court judgments
        judgments_dir = self._data_dir / "court_judgments"
        if judgments_dir.exists():
            judgment_files = [
                f for f in judgments_dir.glob("*.txt")
                if not f.stem.endswith("_facts")
            ]
            if judgment_files:
                datasets.append({
                    "dataset_id": "court_judgments",
                    "name": "Court Judgments",
                    "type": "court_judgment",
                    "record_count": len(judgment_files),
                    "loaded": False,
                    "metadata": {},
                })

        # Forensic image datasets
        image_proc = ForensicImageProcessor(self._data_dir)
        image_labels = {
            "fingerprint": "SOCOFing Fingerprints",
            "wound": "AZH Wound Images",
            "document_forensics": "CEDAR Signature Images",
            "ballistics": "Synthetic Ballistics Images",
            "tool_marks": "Synthetic Tool Mark Images",
        }
        for img_type, label in image_labels.items():
            try:
                images = image_proc.discover_images(img_type)
                if images:
                    datasets.append({
                        "dataset_id": f"forensic_images_{img_type}",
                        "name": label,
                        "type": img_type,
                        "record_count": len(images),
                        "loaded": False,
                        "metadata": {},
                    })
            except Exception:
                pass

        return datasets

    async def generate_and_process_firs(
        self,
        count: int = 10,
        crime_types: Optional[List[str]] = None,
        include_bloodstain: bool = True,
        neo4j_client=None,
    ) -> Dict[str, Any]:
        """Generate synthetic FIRs, save, and process through extraction."""
        # Generate
        firs = await self._fir_gen.generate_batch(count, crime_types, include_bloodstain, neo4j_client=neo4j_client)

        # Save
        fir_dir = self._data_dir / "fir_text"
        await self._fir_gen.save_batch(firs, fir_dir)

        # Process through extraction pipeline
        processed = []
        errors = []
        for fir in firs:
            try:
                result = await self._pipeline.process_text(
                    text=fir["full_text"],
                    source_type="fir",
                    store=True,
                    doc_id=fir["fir_number"],
                )
                processed.append({
                    "fir_number": fir["fir_number"],
                    "entities": result["metadata"].get("entity_count", 0),
                    "relationships": result["metadata"].get("relationship_count", 0),
                })
            except Exception as e:
                errors.append({"fir_number": fir["fir_number"], "error": str(e)})
                logger.error(f"Error processing {fir['fir_number']}: {e}")

        return {
            "generated": len(firs),
            "processed": processed,
            "errors": errors,
            "output_dir": str(fir_dir),
        }

    async def process_court_judgments(
        self, limit: int = 20
    ) -> Dict[str, Any]:
        """Process ingested court judgment texts through extraction into the graph."""
        judgments_dir = self._data_dir / "court_judgments"
        if not judgments_dir.exists():
            return {"processed": [], "errors": [], "total": 0}

        judgment_files = sorted(
            f for f in judgments_dir.glob("*.txt")
            if not f.stem.endswith("_facts")
        )[:limit]

        processed = []
        errors = []
        for jf in judgment_files:
            case_id = jf.stem
            try:
                text = jf.read_text(encoding="utf-8")
                result = await self._pipeline.process_text(
                    text=text,
                    source_type="court_judgment",
                    store=True,
                    doc_id=case_id,
                )
                processed.append({
                    "case_id": case_id,
                    "entities": result["metadata"].get("entity_count", 0),
                    "relationships": result["metadata"].get("relationship_count", 0),
                })
                logger.info(f"Processed judgment {case_id}")
            except Exception as e:
                errors.append({"case_id": case_id, "error": str(e)})
                logger.error(f"Error processing judgment {case_id}: {e}")

        return {"processed": processed, "errors": errors, "total": len(judgment_files)}

    async def process_documents(
        self, doc_type: str, limit: int = 20
    ) -> Dict[str, Any]:
        """Process text documents of any type into the knowledge graph."""
        dir_map = {
            "fir": ("fir_text", "fir"),
            "postmortem": ("postmortem", "postmortem"),
            "lab_report": ("lab_reports", "lab_report"),
            "witness_deposition": ("depositions", "witness_deposition"),
            "soco_report": ("soco", "soco_report"),
            "court_judgment": ("court_judgments", "court_judgment"),
        }
        if doc_type not in dir_map:
            return {"processed": [], "errors": [], "total": 0,
                    "error": f"Unknown doc_type: {doc_type}"}

        dir_name, source_type = dir_map[doc_type]
        docs_dir = self._data_dir / dir_name
        if not docs_dir.exists():
            return {"processed": [], "errors": [], "total": 0}

        doc_files = sorted(
            f for f in docs_dir.glob("*.txt")
            if not f.stem.endswith("_facts")
        )[:limit]

        processed = []
        errors = []
        for df in doc_files:
            doc_id = df.stem
            try:
                text = df.read_text(encoding="utf-8")
                result = await self._pipeline.process_text(
                    text=text,
                    source_type=source_type,
                    store=True,
                    doc_id=doc_id,
                )
                processed.append({
                    "doc_id": doc_id,
                    "entities": result["metadata"].get("entity_count", 0),
                    "relationships": result["metadata"].get("relationship_count", 0),
                })
                logger.info(f"Processed {doc_type} {doc_id}")
            except Exception as e:
                errors.append({"doc_id": doc_id, "error": str(e)})
                logger.error(f"Error processing {doc_type} {doc_id}: {e}")

        return {"processed": processed, "errors": errors, "total": len(doc_files)}

    async def process_bloodstain_data(
        self, limit: int = 5
    ) -> Dict[str, Any]:
        """Process bloodstain experiment data."""
        return await self._bloodstain.process_all(self._pipeline, limit=limit)
