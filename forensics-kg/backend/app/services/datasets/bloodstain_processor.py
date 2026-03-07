import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger


class BloodstainDataProcessor:
    """Processes existing bloodstain experiment data directories."""

    def __init__(self, data_dirs: List[Path]):
        self._data_dirs = data_dirs

    def discover_experiments(self) -> List[Dict[str, Path]]:
        """Find all experiment directories with .txt and .jpg files."""
        experiments = []
        for data_dir in self._data_dirs:
            if not data_dir.exists():
                logger.warning(f"Data dir not found: {data_dir}")
                continue
            for exp_dir in sorted(data_dir.iterdir()):
                if not exp_dir.is_dir() or exp_dir.name.startswith("."):
                    continue
                txt_files = list(exp_dir.glob("*.txt"))
                jpg_files = list(exp_dir.glob("*.jpg")) + list(exp_dir.glob("*.JPG"))
                if txt_files and jpg_files:
                    experiments.append({
                        "experiment_id": exp_dir.name,
                        "metadata_file": txt_files[0],
                        "image_file": jpg_files[0],
                        "directory": exp_dir,
                    })
        logger.info(f"Discovered {len(experiments)} experiments")
        return experiments

    def parse_metadata(self, metadata_file: Path) -> Dict[str, Any]:
        """Parse experiment metadata from text file."""
        try:
            content = metadata_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            content = metadata_file.read_text(encoding="latin-1")

        data: Dict[str, Any] = {
            "experiment_id": metadata_file.stem,
            "category": self._determine_category(metadata_file.stem),
            "raw_text": content,
        }

        lines = content.strip().split("\n")

        # Date (first line)
        if lines:
            data["date"] = lines[0].strip()
        # Description (second line)
        if len(lines) > 1:
            data["description"] = lines[1].strip()

        # Key-value patterns
        kv = {
            "designed_by": r"Designed by:\s*(.+)",
            "performed_by": r"Performed by:\s*(.+)",
            "image_scale": r"Image Scale:\s*(.+)",
            "blood_supply": r"Blood supply\s*=\s*(.+)",
            "target_material": r"Target:\s*(.+)",
            "target_position": r"Target position:\s*(.+)",
            "blood_type": r"Blood Properties:\s*(.+)",
        }
        for key, pattern in kv.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                data[key] = match.group(1).strip()

        # Numeric patterns
        numerics = {
            "room_temp": r"Room Temp\s*=\s*([\d.]+)",
            "room_humidity": r"Room Humidity\s*=\s*([\d.]+)",
            "hematocrit": r"Hematocrit\s*=\s*([\d.]+)",
            "blood_volume": r"Blood Volume\s*=\s*([\d.]+)",
            "height": r"Height[^:]*:\s*([\d.]+)",
            "origin_x": r"x_o\s*=\s*([\d.]+)",
            "origin_y": r"y_o\s*=\s*([\d.]+)",
            "origin_z": r"z_o\s*=\s*([\d.]+)",
        }
        for key, pattern in numerics.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                try:
                    data[key] = float(match.group(1))
                except ValueError:
                    pass

        # HP-specific parameters
        for key, pattern in {
            "dowel_height_station": r"Dowel height station:\s*(\d+)",
            "dowel_angle": r"Dowel Angle:\s*([\d.]+)",
            "rubber_bungees": r"# of Rubber Bungees:\s*(\d+)",
            "moment_arm": r"Moment arm:\s*([\d.]+)",
        }.items():
            match = re.search(pattern, content)
            if match:
                try:
                    data[key] = float(match.group(1))
                except ValueError:
                    pass

        return data

    def _determine_category(self, experiment_id: str) -> str:
        if experiment_id.startswith("C"):
            return "cylinder_experiments"
        elif experiment_id.startswith("HP"):
            return "hockey_puck_experiments"
        return "other"

    async def process_all(
        self, extraction_pipeline, limit: Optional[int] = None
    ) -> Dict[str, Any]:
        """Process experiments through the extraction pipeline."""
        experiments = self.discover_experiments()
        if limit:
            experiments = experiments[:limit]

        results = {"processed": [], "errors": [], "total": len(experiments)}
        for exp in experiments:
            try:
                metadata = self.parse_metadata(exp["metadata_file"])
                image_bytes = exp["image_file"].read_bytes()

                result = await extraction_pipeline.process_image(
                    image_bytes=image_bytes,
                    metadata=metadata,
                    store=True,
                    image_type="bloodstain",
                    doc_id=exp["experiment_id"],
                )
                results["processed"].append({
                    "experiment_id": exp["experiment_id"],
                    "entities": len(result.get("entities", [])),
                    "relationships": len(result.get("relationships", [])),
                })
                logger.info(
                    f"Processed {exp['experiment_id']}: "
                    f"{len(result.get('entities', []))} entities"
                )
            except Exception as e:
                results["errors"].append({
                    "experiment_id": exp["experiment_id"],
                    "error": str(e),
                })
                logger.error(f"Error processing {exp['experiment_id']}: {e}")

        return results
