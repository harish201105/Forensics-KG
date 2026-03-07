"""Tool marks pattern analysis strategy."""

import json
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from app.services.extraction.image_strategies import ImageAnalysisStrategy


TOOL_MARKS_SCHEMA = {
    "type": "object",
    "properties": {
        "mark_type": {
            "type": "string",
            "enum": [
                "striation", "impression", "combination",
                "cut_mark", "pry_mark", "drill_mark", "unknown",
            ],
        },
        "tool_class": {"type": "string"},
        "striation_count": {"type": "integer"},
        "depth_estimate": {"type": "string"},
        "force_estimate": {"type": "string"},
        "description": {"type": "string"},
        "key_features": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "mark_type", "tool_class", "striation_count", "depth_estimate",
        "force_estimate", "description", "key_features",
        "confidence", "uncertainties",
    ],
    "additionalProperties": False,
}

TOOL_MARKS_COMBINED_SCHEMA = {
    "type": "object",
    "properties": {
        "mark_type": {"type": "string"},
        "tool_class": {"type": "string"},
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
        "mark_type", "tool_class", "confidence", "reasoning",
        "key_features", "uncertainties", "relationships",
    ],
    "additionalProperties": False,
}


class ToolMarksStrategy(ImageAnalysisStrategy):
    """Tool marks analysis: striation extraction + depth profiling + GPT-4o vision."""

    def __init__(self, settings):
        self._settings = settings

    def get_schema(self) -> dict:
        return TOOL_MARKS_SCHEMA

    def get_combined_schema(self) -> dict:
        return TOOL_MARKS_COMBINED_SCHEMA

    async def preprocess(self, image_bytes: bytes) -> Dict[str, Any]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"striation_count": 0, "mark_features": {}}

        max_dim = getattr(self._settings, "image_max_size", 1024)
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # Enhance detail
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Edge detection for striations
        edges = cv2.Canny(enhanced, 30, 100)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20, minLineLength=10, maxLineGap=5)
        striation_count = len(lines) if lines is not None else 0

        # Depth profiling via shadow analysis (gradient magnitude)
        grad_x = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(grad_x**2 + grad_y**2)
        max_gradient = float(np.max(gradient_mag))
        mean_gradient = float(np.mean(gradient_mag))

        # Texture analysis
        laplacian_var = float(cv2.Laplacian(enhanced, cv2.CV_64F).var())

        return {
            "striation_count": striation_count,
            "mark_features": {
                "edge_density": float(np.sum(edges > 0) / edges.size),
                "max_gradient": round(max_gradient, 2),
                "mean_gradient": round(mean_gradient, 2),
                "texture_sharpness": round(laplacian_var, 2),
            },
            "image_shape": list(image.shape[:2]),
        }

    async def vision_analysis(self, image_b64: str, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "system_prompt": (
                "You are a forensic tool marks examiner. Analyze the tool mark image "
                "and classify the mark type, identify the probable tool class, "
                "and assess key characteristics. Use AFTE terminology."
            ),
            "user_prompt": (
                "Analyze this tool mark evidence image. Identify:\n"
                "1. Mark type (striation, impression, combination, cut, pry, drill)\n"
                "2. Probable tool class (screwdriver, pliers, bolt cutters, saw, etc.)\n"
                "3. Number of visible striations\n"
                "4. Estimated depth and force\n"
                "5. Key distinguishing features for tool comparison\n"
                "6. Confidence and uncertainties"
            ),
        }

    async def combine(self, cv_results: Dict, vision_results: Dict, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        return {
            "system_prompt": (
                "You are a forensic tool marks examiner. Synthesize the computer vision "
                "striation analysis and AI visual analysis into a comprehensive assessment."
            ),
            "user_prompt": (
                f"AI Visual Analysis:\n{json.dumps(vision_results, indent=2)}\n\n"
                f"CV Striation Analysis:\nCount: {cv_results.get('striation_count', 0)}\n"
                f"Features: {json.dumps(cv_results.get('mark_features', {}), indent=2)}\n\n"
                f"Provide a combined tool marks analysis."
            ),
        }

    def build_extraction_result(self, analysis: Dict, cv_features: Dict, metadata: Optional[Dict]) -> Dict[str, Any]:
        case_id = (metadata or {}).get("case_id", "unknown")
        entities = []
        relationships = []

        tm_id = f"toolmark_{case_id}"
        entities.append({
            "entity_type": "ToolMarkPattern",
            "entity_id": tm_id,
            "properties": {
                "toolmark_id": tm_id,
                "mark_type": analysis.get("mark_type", "unknown"),
                "tool_class": analysis.get("tool_class", "unknown"),
                "striation_count": cv_features.get("striation_count", 0),
                "depth": 0.0,
                "description": analysis.get("reasoning", analysis.get("description", "")),
                "confidence": analysis.get("confidence", 0.5),
            },
        })

        return {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "striation_count": cv_features.get("striation_count", 0),
                "mark_type": analysis.get("mark_type"),
                "tool_class": analysis.get("tool_class"),
                "confidence": analysis.get("confidence"),
            },
        }
