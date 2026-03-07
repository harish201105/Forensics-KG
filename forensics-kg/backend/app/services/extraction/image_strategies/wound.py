"""Wound pattern analysis strategy."""

import json
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

from app.services.extraction.image_strategies import ImageAnalysisStrategy


# --- Wound Analysis Schemas ---

WOUND_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "wound_type": {
            "type": "string",
            "enum": [
                "laceration", "contusion", "abrasion", "incised_wound",
                "stab_wound", "gunshot_entry", "gunshot_exit",
                "burn", "ligature_mark", "bite_mark", "defense_wound", "unknown",
            ],
        },
        "mechanism": {
            "type": "string",
            "enum": [
                "sharp_force", "blunt_force", "gunshot",
                "thermal", "chemical", "asphyxia", "unknown",
            ],
        },
        "estimated_age": {"type": "string"},
        "severity": {"type": "string"},
        "description": {"type": "string"},
        "key_features": {"type": "array", "items": {"type": "string"}},
        "tissue_types_visible": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "wound_type", "mechanism", "estimated_age", "severity",
        "description", "key_features", "tissue_types_visible",
        "confidence", "uncertainties",
    ],
    "additionalProperties": False,
}

WOUND_COMBINED_SCHEMA = {
    "type": "object",
    "properties": {
        "wound_type": {"type": "string"},
        "mechanism": {"type": "string"},
        "estimated_age": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "tissue_analysis": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "tissue_type": {"type": "string"},
                    "color_range": {"type": "string"},
                    "percentage": {"type": "number"},
                    "interpretation": {"type": "string"},
                },
                "required": ["tissue_type", "color_range", "percentage", "interpretation"],
                "additionalProperties": False,
            },
        },
        "measurements": {
            "type": "object",
            "properties": {
                "estimated_area_px": {"type": "number"},
                "wound_ratio": {"type": "number"},
                "color_variance": {"type": "number"},
            },
            "required": ["estimated_area_px", "wound_ratio", "color_variance"],
            "additionalProperties": False,
        },
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
        "wound_type", "mechanism", "estimated_age", "confidence",
        "reasoning", "tissue_analysis", "measurements",
        "key_features", "uncertainties", "relationships",
    ],
    "additionalProperties": False,
}


class WoundStrategy(ImageAnalysisStrategy):
    """Wound analysis: color segmentation + area measurement + GPT-4o vision."""

    def __init__(self, settings):
        self._settings = settings

    def get_schema(self) -> dict:
        return WOUND_ANALYSIS_SCHEMA

    def get_combined_schema(self) -> dict:
        return WOUND_COMBINED_SCHEMA

    async def preprocess(self, image_bytes: bytes) -> Dict[str, Any]:
        """OpenCV: color segmentation for tissue types, area measurement."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"wound_detected": False, "tissue_segments": [], "measurements": {}}

        max_dim = getattr(self._settings, "image_max_size", 1024)
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Color-based tissue segmentation
        tissue_segments = []

        # Red/blood regions (wound area)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        red_pct = float(np.sum(mask_red > 0) / mask_red.size)
        tissue_segments.append({
            "tissue_type": "hemorrhagic",
            "color": "red",
            "percentage": round(red_pct * 100, 2),
            "pixel_count": int(np.sum(mask_red > 0)),
        })

        # Dark regions (necrotic/bruised)
        l_channel = lab[:, :, 0]
        dark_mask = (l_channel < 60).astype(np.uint8) * 255
        dark_pct = float(np.sum(dark_mask > 0) / dark_mask.size)
        tissue_segments.append({
            "tissue_type": "necrotic_or_bruised",
            "color": "dark",
            "percentage": round(dark_pct * 100, 2),
            "pixel_count": int(np.sum(dark_mask > 0)),
        })

        # Yellow/green regions (healing/infection)
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([40, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        yellow_pct = float(np.sum(mask_yellow > 0) / mask_yellow.size)
        tissue_segments.append({
            "tissue_type": "healing_or_infected",
            "color": "yellow_green",
            "percentage": round(yellow_pct * 100, 2),
            "pixel_count": int(np.sum(mask_yellow > 0)),
        })

        # Wound area estimation (combine red + dark as wound)
        wound_mask = cv2.bitwise_or(mask_red, dark_mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        wound_mask = cv2.morphologyEx(wound_mask, cv2.MORPH_CLOSE, kernel)
        wound_area = int(np.sum(wound_mask > 0))
        wound_ratio = float(wound_area / wound_mask.size)

        # Color variance in wound region
        wound_pixels = image[wound_mask > 0]
        color_variance = float(np.var(np.asarray(wound_pixels, dtype=np.float64))) if len(wound_pixels) > 0 else 0

        measurements = {
            "estimated_area_px": wound_area,
            "wound_ratio": round(wound_ratio, 4),
            "color_variance": round(color_variance, 2),
            "image_shape": list(image.shape[:2]),
        }

        return {
            "wound_detected": wound_ratio > 0.01,
            "tissue_segments": tissue_segments,
            "measurements": measurements,
        }

    async def vision_analysis(
        self, image_b64: str, cv_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "system_prompt": (
                "You are a forensic pathology and wound analysis expert. "
                "Analyze the wound image and classify the wound type, mechanism, "
                "estimate age, and identify key features. Use standard forensic "
                "wound classification terminology."
            ),
            "user_prompt": (
                "Analyze this forensic wound image. Identify:\n"
                "1. The wound type (laceration, contusion, abrasion, incised, stab, gunshot, etc.)\n"
                "2. The probable mechanism (sharp force, blunt force, gunshot, thermal, etc.)\n"
                "3. Estimated age of the wound (fresh, hours, days)\n"
                "4. Severity assessment\n"
                "5. Types of tissue visible (hemorrhagic, necrotic, granulation, etc.)\n"
                "6. Key diagnostic features\n"
                "7. Your confidence level and uncertainties"
            ),
        }

    async def combine(
        self,
        cv_results: Dict[str, Any],
        vision_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        tissue_summary = "\n".join(
            f"  {t['tissue_type']}: {t['percentage']}%"
            for t in cv_results.get("tissue_segments", [])
        )
        measurements = cv_results.get("measurements", {})
        return {
            "system_prompt": (
                "You are a forensic wound analysis expert. "
                "Synthesize the computer vision tissue segmentation and AI visual analysis "
                "into a comprehensive wound assessment."
            ),
            "user_prompt": (
                f"AI Visual Analysis:\n{json.dumps(vision_results, indent=2)}\n\n"
                f"Computer Vision Tissue Segmentation:\n{tissue_summary}\n"
                f"Measurements: {json.dumps(measurements, indent=2)}\n\n"
                f"Metadata:\n{json.dumps(metadata or {}, indent=2)}\n\n"
                f"Provide a combined wound analysis with type, mechanism, "
                f"tissue analysis, and age estimation."
            ),
        }

    def build_extraction_result(
        self, analysis: Dict, cv_features: Dict, metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Convert analysis into entities + relationships."""
        case_id = (metadata or {}).get("case_id", "unknown")
        entities = []
        relationships = []

        wound_id = f"wound_{case_id}"
        entities.append({
            "entity_type": "WoundPattern",
            "entity_id": wound_id,
            "properties": {
                "wound_id": wound_id,
                "wound_type": analysis.get("wound_type", "unknown"),
                "mechanism": analysis.get("mechanism", "unknown"),
                "estimated_age": analysis.get("estimated_age", ""),
                "area": cv_features.get("measurements", {}).get("estimated_area_px", 0),
                "description": analysis.get("reasoning", analysis.get("description", "")),
                "confidence": analysis.get("confidence", 0.5),
            },
        })

        # Tissue analysis entities
        for i, t in enumerate(analysis.get("tissue_analysis", [])):
            t_id = f"tissue_{case_id}_{i}"
            entities.append({
                "entity_type": "TissueAnalysis",
                "entity_id": t_id,
                "properties": {
                    "tissue_id": t_id,
                    "tissue_type": t.get("tissue_type", "unknown"),
                    "color_profile": t.get("color_range", ""),
                    "percentage": t.get("percentage", 0),
                    "description": t.get("interpretation", ""),
                },
            })
            relationships.append({
                "source_entity_id": t_id,
                "target_entity_id": wound_id,
                "relationship_type": "TISSUE_TYPE_OF",
                "confidence": 0.85,
                "evidence_span": "Wound tissue analysis",
            })

        return {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "wound_detected": cv_features.get("wound_detected", False),
                "wound_type": analysis.get("wound_type"),
                "mechanism": analysis.get("mechanism"),
                "confidence": analysis.get("confidence"),
            },
        }
