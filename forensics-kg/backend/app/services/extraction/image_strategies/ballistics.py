"""Ballistics pattern analysis strategy."""

import json
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from app.services.extraction.image_strategies import ImageAnalysisStrategy


BALLISTICS_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern_type": {
            "type": "string",
            "enum": [
                "bullet", "cartridge_case", "rifling_marks",
                "firing_pin_impression", "breech_face_marks",
                "extractor_marks", "ejector_marks", "unknown",
            ],
        },
        "caliber_estimate": {"type": "string"},
        "rifling_characteristics": {"type": "string"},
        "firearm_class": {"type": "string"},
        "description": {"type": "string"},
        "key_features": {"type": "array", "items": {"type": "string"}},
        "striation_count": {"type": "integer"},
        "confidence": {"type": "number"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "pattern_type", "caliber_estimate", "rifling_characteristics",
        "firearm_class", "description", "key_features", "striation_count",
        "confidence", "uncertainties",
    ],
    "additionalProperties": False,
}

BALLISTICS_COMBINED_SCHEMA = {
    "type": "object",
    "properties": {
        "pattern_type": {"type": "string"},
        "caliber_estimate": {"type": "string"},
        "firearm_class": {"type": "string"},
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
        "pattern_type", "caliber_estimate", "firearm_class",
        "confidence", "reasoning", "key_features", "uncertainties", "relationships",
    ],
    "additionalProperties": False,
}


class BallisticsStrategy(ImageAnalysisStrategy):
    """Ballistics analysis: edge detection for striations + GPT-4o vision."""

    def __init__(self, settings):
        self._settings = settings

    def get_schema(self) -> dict:
        return BALLISTICS_ANALYSIS_SCHEMA

    def get_combined_schema(self) -> dict:
        return BALLISTICS_COMBINED_SCHEMA

    async def preprocess(self, image_bytes: bytes) -> Dict[str, Any]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"striation_count": 0, "edge_features": {}}

        max_dim = getattr(self._settings, "image_max_size", 1024)
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # CLAHE for detail enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Edge detection for striations
        edges = cv2.Canny(enhanced, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=20, maxLineGap=5)
        striation_count = len(lines) if lines is not None else 0

        # Orientation analysis
        orientations = []
        if lines is not None:
            for line in lines:
                pts = line.flatten().tolist()
                x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
                angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                orientations.append(angle)

        return {
            "striation_count": striation_count,
            "edge_features": {
                "edge_density": float(np.sum(edges > 0) / edges.size),
                "dominant_orientation": float(np.median(orientations)) if orientations else 0,
                "orientation_std": float(np.std(orientations)) if orientations else 0,
            },
            "image_shape": list(image.shape[:2]),
        }

    async def vision_analysis(self, image_b64: str, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "system_prompt": (
                "You are a forensic ballistics expert (firearms examiner). "
                "Analyze the ballistics evidence image and classify the pattern, "
                "estimate caliber, identify rifling characteristics, and determine "
                "the probable firearm class. Use AFTE terminology."
            ),
            "user_prompt": (
                "Analyze this ballistics evidence image. Identify:\n"
                "1. Type of evidence (bullet, cartridge case, rifling marks, etc.)\n"
                "2. Estimated caliber\n"
                "3. Rifling characteristics (number of lands/grooves, twist direction)\n"
                "4. Probable firearm class\n"
                "5. Number of visible striations\n"
                "6. Key distinguishing features\n"
                "7. Confidence and uncertainties"
            ),
        }

    async def combine(self, cv_results: Dict, vision_results: Dict, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        return {
            "system_prompt": (
                "You are a forensic ballistics expert. Synthesize the computer vision "
                "edge analysis and AI visual analysis into a comprehensive ballistics assessment."
            ),
            "user_prompt": (
                f"AI Visual Analysis:\n{json.dumps(vision_results, indent=2)}\n\n"
                f"CV Edge Features:\nStriations: {cv_results.get('striation_count', 0)}\n"
                f"Edge details: {json.dumps(cv_results.get('edge_features', {}), indent=2)}\n\n"
                f"Provide a combined ballistics analysis."
            ),
        }

    def build_extraction_result(self, analysis: Dict, cv_features: Dict, metadata: Optional[Dict]) -> Dict[str, Any]:
        case_id = (metadata or {}).get("case_id", "unknown")
        entities = []
        relationships = []

        b_id = f"ballistics_{case_id}"
        entities.append({
            "entity_type": "BallisticsPattern",
            "entity_id": b_id,
            "properties": {
                "ballistics_id": b_id,
                "pattern_type": analysis.get("pattern_type", "unknown"),
                "caliber": analysis.get("caliber_estimate", ""),
                "rifling_characteristics": analysis.get("rifling_characteristics", ""),
                "description": analysis.get("reasoning", analysis.get("description", "")),
                "confidence": analysis.get("confidence", 0.5),
            },
        })

        return {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "striation_count": cv_features.get("striation_count", 0),
                "pattern_type": analysis.get("pattern_type"),
                "caliber": analysis.get("caliber_estimate"),
                "confidence": analysis.get("confidence"),
            },
        }
