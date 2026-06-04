"""Fingerprint pattern analysis strategy."""

import json
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

from app.services.extraction.image_strategies import ImageAnalysisStrategy


# --- Fingerprint Analysis Schemas ---

FINGERPRINT_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {
            "type": "string",
            "enum": [
                "loop_ulnar", "loop_radial", "whorl_plain",
                "whorl_central_pocket", "whorl_double_loop", "whorl_accidental",
                "arch_plain", "arch_tented", "unknown",
            ],
        },
        "quality_score": {"type": "number"},
        "ridge_count": {"type": "integer"},
        "minutiae_count": {"type": "integer"},
        "hand": {"type": "string", "enum": ["left", "right", "unknown"]},
        "finger_position": {
            "type": "string",
            "enum": ["thumb", "index", "middle", "ring", "little", "unknown"],
        },
        "description": {"type": "string"},
        "key_features": {"type": "array", "items": {"type": "string"}},
        "suitability_for_comparison": {"type": "string"},
        "confidence": {"type": "number"},
        "uncertainties": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "classification", "quality_score", "ridge_count", "minutiae_count",
        "hand", "finger_position",
        "description", "key_features", "suitability_for_comparison",
        "confidence", "uncertainties",
    ],
    "additionalProperties": False,
}

FINGERPRINT_COMBINED_SCHEMA = {
    "type": "object",
    "properties": {
        "classification": {"type": "string"},
        "quality_score": {"type": "number"},
        "hand": {"type": "string"},
        "finger_position": {"type": "string"},
        "confidence": {"type": "number"},
        "reasoning": {"type": "string"},
        "minutiae_details": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "x": {"type": "number"},
                    "y": {"type": "number"},
                    "description": {"type": "string"},
                },
                "required": ["type", "x", "y", "description"],
                "additionalProperties": False,
            },
        },
        "ridge_characteristics": {"type": "string"},
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
        "classification", "quality_score", "hand", "finger_position",
        "confidence", "reasoning",
        "minutiae_details", "ridge_characteristics", "key_features",
        "uncertainties", "relationships",
    ],
    "additionalProperties": False,
}


class FingerprintStrategy(ImageAnalysisStrategy):
    """Fingerprint analysis: ridge enhancement + minutiae detection + GPT-4o vision."""

    def __init__(self, settings):
        self._settings = settings

    def get_schema(self) -> dict:
        return FINGERPRINT_ANALYSIS_SCHEMA

    def get_combined_schema(self) -> dict:
        return FINGERPRINT_COMBINED_SCHEMA

    async def preprocess(self, image_bytes: bytes) -> Dict[str, Any]:
        """OpenCV: ridge enhancement, thinning, minutiae detection."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            return {"minutiae_count": 0, "ridge_features": {}, "quality_metrics": {}}

        # Resize
        max_dim = getattr(self._settings, "image_max_size", 1024)
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        # CLAHE enhancement for ridges
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(image)

        # Gabor filter bank for ridge orientation
        orientations = []
        for theta in np.arange(0, np.pi, np.pi / 8):
            kernel = cv2.getGaborKernel(
                (21, 21), sigma=4.0, theta=float(theta),
                lambd=10.0, gamma=0.5, psi=0
            )
            filtered = cv2.filter2D(enhanced, cv2.CV_64F, kernel)
            orientations.append(np.mean(np.abs(filtered)))

        dominant_orientation = float(np.argmax(orientations) * (180 / 8))

        # Adaptive threshold for binarization
        binary = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 15, 5
        )

        # Morphological thinning approximation
        kernel = np.ones((3, 3), np.uint8)
        thinned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        thinned = cv2.morphologyEx(thinned, cv2.MORPH_CLOSE, kernel)

        # Minutiae detection via Harris corners on thinned image
        corners = cv2.cornerHarris(thinned.astype(np.float32), 3, 3, 0.04)
        corners = cv2.dilate(corners, np.ones((3, 3), np.uint8))
        threshold = 0.01 * corners.max() if corners.max() > 0 else 0
        minutiae_points = np.argwhere(corners > threshold)

        # Cluster nearby points
        minutiae_clustered = self._cluster_minutiae(minutiae_points, min_distance=10)

        # Quality assessment
        quality = self._assess_quality(enhanced, binary)

        return {
            "minutiae_count": len(minutiae_clustered),
            "minutiae_points": [
                {"x": float(p[1]), "y": float(p[0])} for p in minutiae_clustered[:50]
            ],
            "ridge_features": {
                "dominant_orientation": dominant_orientation,
                "orientation_strengths": [float(o) for o in orientations],
            },
            "quality_metrics": quality,
            "image_shape": list(image.shape[:2]),
        }

    async def vision_analysis(
        self, image_b64: str, cv_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        return {
            "system_prompt": (
                "You are a forensic fingerprint analysis expert (latent print examiner). "
                "Analyze the fingerprint image and classify the pattern, identify key "
                "minutiae features, and assess the print quality for comparison purposes. "
                "Use ACE-V methodology terminology."
            ),
            "user_prompt": (
                "Analyze this fingerprint image. Identify:\n"
                "1. The fingerprint classification (loop/whorl/arch subtype)\n"
                "2. Estimated ridge count\n"
                "3. Number and types of minutiae (bifurcations, ridge endings, dots)\n"
                "4. Overall quality score (0-1) and suitability for comparison\n"
                "5. Your best estimate of the hand (left/right) and finger position "
                "(thumb/index/middle/ring/little); use 'unknown' if not determinable\n"
                "6. Key distinctive features\n"
                "7. Your confidence level and any uncertainties"
            ),
        }

    async def combine(
        self,
        cv_results: Dict[str, Any],
        vision_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cv_summary = (
            f"CV detected {cv_results.get('minutiae_count', 0)} minutiae points. "
            f"Dominant ridge orientation: {cv_results.get('ridge_features', {}).get('dominant_orientation', 'N/A')}°. "
            f"Quality: {json.dumps(cv_results.get('quality_metrics', {}))}"
        )
        return {
            "system_prompt": (
                "You are a forensic fingerprint analysis expert. "
                "Synthesize the computer vision minutiae detection and AI visual analysis "
                "into a comprehensive fingerprint assessment."
            ),
            "user_prompt": (
                f"AI Visual Analysis:\n{json.dumps(vision_results, indent=2)}\n\n"
                f"Computer Vision Features:\n{cv_summary}\n\n"
                f"Metadata:\n{json.dumps(metadata or {}, indent=2)}\n\n"
                f"Provide a combined fingerprint analysis with classification, "
                f"minutiae details, and suitability for comparison."
            ),
        }

    def build_extraction_result(
        self, analysis: Dict, cv_features: Dict, metadata: Optional[Dict]
    ) -> Dict[str, Any]:
        """Convert analysis into entities + relationships."""
        case_id = (metadata or {}).get("case_id", "unknown")
        entities = []
        relationships = []

        pattern_id = f"fp_pattern_{case_id}"
        entities.append({
            "entity_type": "FingerprintPattern",
            "entity_id": pattern_id,
            "properties": {
                "pattern_id": pattern_id,
                "classification": analysis.get("classification", "unknown"),
                "ridge_count": analysis.get("ridge_count", 0),
                "quality_score": analysis.get("quality_score", 0),
                "hand": analysis.get("hand", "unknown"),
                "finger_position": analysis.get("finger_position", "unknown"),
                "description": analysis.get("description", ""),
                "confidence": analysis.get("confidence", 0.5),
            },
        })

        # Minutiae entities
        for i, m in enumerate(analysis.get("minutiae_details", [])[:20]):
            m_id = f"minutiae_{case_id}_{i}"
            entities.append({
                "entity_type": "MinutiaePoint",
                "entity_id": m_id,
                "properties": {
                    "minutiae_id": m_id,
                    "minutiae_type": m.get("type", "unknown"),
                    "x": m.get("x", 0),
                    "y": m.get("y", 0),
                    "angle": 0.0,
                    "quality": analysis.get("quality_score", 0.5),
                },
            })
            relationships.append({
                "source_entity_id": pattern_id,
                "target_entity_id": m_id,
                "relationship_type": "HAS_MINUTIAE",
                "confidence": 0.9,
                "evidence_span": "Fingerprint analysis",
            })

        return {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "cv_minutiae_detected": cv_features.get("minutiae_count", 0),
                "classification": analysis.get("classification"),
                "quality_score": analysis.get("quality_score"),
                "confidence": analysis.get("confidence"),
            },
        }

    def _cluster_minutiae(self, points: np.ndarray, min_distance: int = 10) -> List:
        """Cluster nearby minutiae points to reduce noise."""
        if len(points) == 0:
            return []
        clustered = [points[0]]
        for p in points[1:]:
            distances = [np.sqrt((p[0] - c[0])**2 + (p[1] - c[1])**2) for c in clustered]
            if min(distances) >= min_distance:
                clustered.append(p)
        return clustered

    def _assess_quality(self, enhanced: np.ndarray, binary: np.ndarray) -> Dict[str, float]:
        """Assess fingerprint image quality."""
        # Contrast
        contrast = float(np.std(enhanced))
        # Ridge clarity (ratio of white to total in binary)
        ridge_ratio = float(np.sum(binary > 0) / binary.size)
        # Noise estimate via Laplacian variance
        laplacian_var = float(cv2.Laplacian(enhanced, cv2.CV_64F).var())
        # Overall quality score (heuristic)
        quality_score = min(1.0, (contrast / 80) * 0.4 + (1 - abs(ridge_ratio - 0.3)) * 0.3 + min(laplacian_var / 1000, 1) * 0.3)

        return {
            "contrast": round(contrast, 2),
            "ridge_ratio": round(ridge_ratio, 4),
            "sharpness": round(laplacian_var, 2),
            "overall_quality": round(quality_score, 3),
        }
