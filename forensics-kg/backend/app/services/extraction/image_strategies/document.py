"""Document forensics analysis strategy (handwriting, ink, forgery)."""

import json
import cv2
import numpy as np
from typing import Dict, List, Any, Optional

from app.services.extraction.image_strategies import ImageAnalysisStrategy


DOCUMENT_FORENSICS_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {
            "type": "string",
            "enum": [
                "handwritten_note", "printed_document", "check",
                "signature", "identity_document", "mixed", "unknown",
            ],
        },
        "forgery_indicators": {"type": "array", "items": {"type": "string"}},
        "handwriting_features": {
            "type": "object",
            "properties": {
                "slant": {"type": "string"},
                "baseline": {"type": "string"},
                "pressure": {"type": "string"},
                "letter_spacing": {"type": "string"},
                "word_spacing": {"type": "string"},
            },
            "required": ["slant", "baseline", "pressure", "letter_spacing", "word_spacing"],
            "additionalProperties": False,
        },
        "ink_characteristics": {"type": "string"},
        "paper_characteristics": {"type": "string"},
        "description": {"type": "string"},
        "key_features": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "document_type", "forgery_indicators", "handwriting_features",
        "ink_characteristics", "paper_characteristics", "description",
        "key_features", "confidence", "uncertainties",
    ],
    "additionalProperties": False,
}

DOCUMENT_COMBINED_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {"type": "string"},
        "is_forged": {"type": "boolean"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "forgery_indicators": {"type": "array", "items": {"type": "string"}},
        "handwriting_features": {
            "type": "object",
            "properties": {
                "slant": {"type": "string"},
                "baseline": {"type": "string"},
                "pressure": {"type": "string"},
                "consistency": {"type": "string"},
            },
            "required": ["slant", "baseline", "pressure", "consistency"],
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
        "document_type", "is_forged", "confidence", "reasoning",
        "forgery_indicators", "handwriting_features",
        "key_features", "uncertainties", "relationships",
    ],
    "additionalProperties": False,
}


class DocumentForensicsStrategy(ImageAnalysisStrategy):
    """Document forensics: ink/baseline/slant analysis + GPT-4o vision."""

    def __init__(self, settings):
        self._settings = settings

    def get_schema(self) -> dict:
        return DOCUMENT_FORENSICS_SCHEMA

    def get_combined_schema(self) -> dict:
        return DOCUMENT_COMBINED_SCHEMA

    async def preprocess(self, image_bytes: bytes) -> Dict[str, Any]:
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"text_detected": False, "writing_features": {}}

        max_dim = getattr(self._settings, "image_max_size", 1024)
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Binarize for text detection
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        # Baseline detection via horizontal projection
        h_proj = np.sum(binary, axis=1)
        text_rows = np.where(h_proj > (w * 0.05))[0]
        baseline_count = 0
        if len(text_rows) > 0:
            # Count transitions from text to non-text
            diffs = np.diff(text_rows)
            baseline_count = int(np.sum(diffs > 5)) + 1

        # Slant estimation via Hough lines on text
        edges = cv2.Canny(binary, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=15, maxLineGap=3)
        angles = []
        if lines is not None:
            for line in lines:
                pts = line.flatten().tolist()
                x1, y1, x2, y2 = pts[0], pts[1], pts[2], pts[3]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)

        avg_slant = float(np.mean(angles)) if angles else 0
        slant_std = float(np.std(angles)) if angles else 0

        # Ink density analysis (color channels)
        ink_pixels = image[binary > 0]
        ink_color_mean = list(np.mean(np.asarray(ink_pixels, dtype=np.float64), axis=0)) if len(ink_pixels) > 0 else [0, 0, 0]
        ink_color_std = list(np.std(np.asarray(ink_pixels, dtype=np.float64), axis=0)) if len(ink_pixels) > 0 else [0, 0, 0]

        return {
            "text_detected": len(text_rows) > 0,
            "writing_features": {
                "baseline_count": baseline_count,
                "avg_slant": round(avg_slant, 2),
                "slant_consistency": round(slant_std, 2),
                "text_density": round(float(np.sum(binary > 0) / binary.size), 4),
            },
            "ink_analysis": {
                "mean_color_bgr": [round(c, 1) for c in ink_color_mean],
                "color_variance_bgr": [round(c, 1) for c in ink_color_std],
            },
            "image_shape": list(image.shape[:2]),
        }

    async def vision_analysis(self, image_b64: str, cv_results: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "system_prompt": (
                "You are a forensic document examiner (questioned document expert). "
                "Analyze the document image for handwriting characteristics, "
                "forgery indicators, ink properties, and document authenticity. "
                "Use standard ASTM document examination terminology."
            ),
            "user_prompt": (
                "Analyze this forensic document image. Identify:\n"
                "1. Document type (handwritten, printed, check, signature, etc.)\n"
                "2. Any forgery indicators (tremor, patching, overwriting, etc.)\n"
                "3. Handwriting features (slant, baseline, pressure, spacing)\n"
                "4. Ink characteristics (color, consistency, potential age)\n"
                "5. Paper characteristics if visible\n"
                "6. Key features for comparison\n"
                "7. Confidence and uncertainties"
            ),
        }

    async def combine(self, cv_results: Dict, vision_results: Dict, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        return {
            "system_prompt": (
                "You are a forensic document examiner. Synthesize the computer vision "
                "writing analysis and AI visual analysis into a comprehensive document assessment."
            ),
            "user_prompt": (
                f"AI Visual Analysis:\n{json.dumps(vision_results, indent=2)}\n\n"
                f"CV Writing Features:\n{json.dumps(cv_results.get('writing_features', {}), indent=2)}\n"
                f"CV Ink Analysis:\n{json.dumps(cv_results.get('ink_analysis', {}), indent=2)}\n\n"
                f"Provide a combined document forensics analysis."
            ),
        }

    def build_extraction_result(self, analysis: Dict, cv_features: Dict, metadata: Optional[Dict]) -> Dict[str, Any]:
        case_id = (metadata or {}).get("case_id", "unknown")
        entities = []
        relationships = []

        # Handwriting feature entity
        hw_id = f"handwriting_{case_id}"
        hw_features = analysis.get("handwriting_features", {})
        entities.append({
            "entity_type": "HandwritingFeature",
            "entity_id": hw_id,
            "properties": {
                "feature_id": hw_id,
                "feature_type": analysis.get("document_type", "unknown"),
                "baseline": hw_features.get("baseline", ""),
                "slant": hw_features.get("slant", ""),
                "pressure": hw_features.get("pressure", ""),
                "description": analysis.get("reasoning", analysis.get("description", "")),
            },
        })

        # Forgery indicators
        for i, indicator in enumerate(analysis.get("forgery_indicators", [])):
            f_id = f"forgery_{case_id}_{i}"
            entities.append({
                "entity_type": "ForgeryIndicator",
                "entity_id": f_id,
                "properties": {
                    "forgery_id": f_id,
                    "indicator_type": indicator,
                    "confidence": analysis.get("confidence", 0.5),
                    "description": indicator,
                },
            })
            relationships.append({
                "source_entity_id": f_id,
                "target_entity_id": hw_id,
                "relationship_type": "FORGED_ON",
                "confidence": analysis.get("confidence", 0.5),
                "evidence_span": "Document forensics analysis",
            })

        return {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "document_type": analysis.get("document_type"),
                "is_forged": analysis.get("is_forged", False),
                "confidence": analysis.get("confidence"),
            },
        }
