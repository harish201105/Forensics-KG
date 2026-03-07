"""Generic batch processor for forensic images of any type.

Discovers images on disk, loads co-located metadata (gold JSON for synthetic
images, directory structure for downloaded datasets), and processes them through
the extraction pipeline with the correct ``image_type`` and ``doc_id``.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

# Maps image_type to directory layout and metadata source.
IMAGE_TYPE_CONFIGS: Dict[str, Dict[str, Any]] = {
    "fingerprint": {
        "image_dir": "images/fingerprints/socofing",
        "extensions": ["*.BMP", "*.bmp", "*.png", "*.jpg"],
        "gold_dir": None,
        "recurse": True,
    },
    "wound": {
        "image_dir": "images/wounds/azh",
        "extensions": ["*.jpg", "*.jpeg", "*.png"],
        "gold_dir": None,
        "recurse": True,
    },
    "document_forensics": {
        "image_dir": "images/document_forensics/cedar",
        "extensions": ["*.png", "*.jpg"],
        "gold_dir": None,
        "recurse": True,
    },
    "ballistics": {
        "image_dir": "images/ballistics",
        "extensions": ["*.png"],
        "gold_dir": "gold_ballistics",
        "recurse": False,
    },
    "tool_marks": {
        "image_dir": "images/tool_marks",
        "extensions": ["*.png"],
        "gold_dir": "gold_toolmarks",
        "recurse": False,
    },
}


class ForensicImageProcessor:
    """Batch-process forensic images into the knowledge graph."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    def discover_images(
        self, image_type: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Find all image files for a given forensic image type.

        Returns a list of dicts with keys: image_path, image_id, metadata,
        image_type.
        """
        config = IMAGE_TYPE_CONFIGS.get(image_type)
        if config is None:
            raise ValueError(
                f"Unknown image type '{image_type}'. "
                f"Supported: {list(IMAGE_TYPE_CONFIGS.keys())}"
            )

        image_dir = self._data_dir / config["image_dir"]
        if not image_dir.exists():
            logger.warning(f"Image directory not found: {image_dir}")
            return []

        gold_dir = (
            self._data_dir / config["gold_dir"] if config["gold_dir"] else None
        )

        # Collect image files
        glob_method = image_dir.rglob if config["recurse"] else image_dir.glob
        image_files: List[Path] = []
        for ext in config["extensions"]:
            image_files.extend(glob_method(ext))

        image_files = sorted(set(image_files))
        if limit:
            image_files = image_files[:limit]

        results: List[Dict[str, Any]] = []
        for img_path in image_files:
            image_id = img_path.stem
            metadata: Dict[str, Any] = {"source_file": img_path.name}

            if gold_dir:
                gold_path = gold_dir / f"{image_id}.json"
                if gold_path.exists():
                    try:
                        gold_data = json.loads(
                            gold_path.read_text(encoding="utf-8")
                        )
                        metadata.update(gold_data)
                    except Exception as e:
                        logger.warning(f"Failed to parse gold JSON {gold_path}: {e}")
            else:
                rel_parts = img_path.relative_to(image_dir).parts
                if len(rel_parts) > 1:
                    metadata["subdirectory"] = "/".join(rel_parts[:-1])
                if len(rel_parts) >= 3:
                    metadata["split"] = rel_parts[0]
                    metadata["category"] = rel_parts[1]
                elif len(rel_parts) >= 2:
                    metadata["category"] = rel_parts[0]

            results.append({
                "image_path": img_path,
                "image_id": image_id,
                "metadata": metadata,
                "image_type": image_type,
            })

        logger.info(f"Discovered {len(results)} {image_type} images in {image_dir}")
        return results

    async def process_all(
        self,
        extraction_pipeline: Any,
        image_type: str,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process all discovered images through the extraction pipeline."""
        images = self.discover_images(image_type, limit=limit)

        results: Dict[str, Any] = {
            "processed": [],
            "errors": [],
            "total": len(images),
            "image_type": image_type,
        }

        for img_info in images:
            try:
                image_bytes = img_info["image_path"].read_bytes()

                result = await extraction_pipeline.process_image(
                    image_bytes=image_bytes,
                    metadata=img_info["metadata"],
                    store=True,
                    image_type=image_type,
                    doc_id=img_info["image_id"],
                )

                results["processed"].append({
                    "image_id": img_info["image_id"],
                    "entities": len(result.get("entities", [])),
                    "relationships": len(result.get("relationships", [])),
                })
                logger.info(
                    f"Processed {image_type} image {img_info['image_id']}: "
                    f"{len(result.get('entities', []))} entities"
                )
            except Exception as e:
                results["errors"].append({
                    "image_id": img_info["image_id"],
                    "error": str(e),
                })
                logger.error(
                    f"Error processing {image_type} {img_info['image_id']}: {e}"
                )

        return results
