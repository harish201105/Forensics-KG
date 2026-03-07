"""Bloodstain pattern analysis strategy — refactored from original ImageExtractor."""

import json
import cv2
import numpy as np
from typing import Dict, List, Any, Optional
from loguru import logger

from app.services.extraction.image_strategies import ImageAnalysisStrategy
from app.services.extraction.schemas import (
    BLOODSTAIN_IMAGE_ANALYSIS_SCHEMA,
    BLOODSTAIN_COMBINED_ANALYSIS_SCHEMA,
)


class BloodstainStrategy(ImageAnalysisStrategy):
    """Bloodstain pattern analysis: CLAHE + Otsu segmentation + GPT-4o vision."""

    def __init__(self, settings):
        self._settings = settings

    def get_schema(self) -> dict:
        return BLOODSTAIN_IMAGE_ANALYSIS_SCHEMA

    def get_combined_schema(self) -> dict:
        return BLOODSTAIN_COMBINED_ANALYSIS_SCHEMA

    async def preprocess(self, image_bytes: bytes) -> Dict[str, Any]:
        """Traditional CV pipeline: CLAHE, Otsu segmentation, feature extraction."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return {"total_stains": 0, "stain_features": [], "spatial_features": {}}

        max_dim = self._settings.image_max_size
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image = cv2.resize(
                image, (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_AREA,
            )

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # CLAHE enhancement
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.merge([l_ch, a_ch, b_ch])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)

        # Segmentation: Otsu + morphological ops
        gray = cv2.cvtColor(enhanced, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        binary = cv2.bitwise_not(binary)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = self._settings.min_stain_area
        max_area = self._settings.max_stain_area
        stain_features = []
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                features = self._extract_geometric_features(contour)
                features["stain_id"] = f"stain_{i}"  # will be prefixed in build_extraction_result
                M = cv2.moments(contour)
                if M["m00"] > 0:
                    features["center_x"] = float(M["m10"] / M["m00"])
                    features["center_y"] = float(M["m01"] / M["m00"])
                stain_features.append(features)

        spatial = self._extract_spatial_features(stain_features, image.shape[:2])

        return {
            "total_stains": len(stain_features),
            "stain_features": stain_features,
            "spatial_features": spatial,
            "image_shape": list(image.shape[:2]),
        }

    async def vision_analysis(
        self, image_b64: str, cv_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        # This will be called by the orchestrator which passes the openai_client
        # The actual call is handled in ImageExtractor.analyze()
        return {
            "system_prompt": (
                "You are a forensic bloodstain pattern analysis expert. "
                "Analyze the bloodstain image and classify the pattern type, "
                "impact mechanism, and key features. Be precise and evidence-based. "
                "Base your analysis on the International Association for Identification "
                "(IAI) terminology for bloodstain pattern analysis."
            ),
            "user_prompt": (
                "Analyze this bloodstain pattern image. Identify:\n"
                "1. The bloodstain pattern type (from the standard BPA classifications)\n"
                "2. The probable impact mechanism that created these stains\n"
                "3. Key visual features (shape, directionality, distribution)\n"
                "4. Spatial distribution characteristics\n"
                "5. Estimated number of individual stains visible\n"
                "6. Your confidence level and any uncertainties"
            ),
        }

    async def combine(
        self,
        cv_results: Dict[str, Any],
        vision_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # Build the combined prompt context
        stain_summary = ""
        stains = cv_results.get("stain_features", [])
        if stains:
            areas = [s["area"] for s in stains]
            circs = [s["circularity"] for s in stains]
            stain_summary = (
                f"CV detected {len(stains)} stains. "
                f"Area range: {min(areas):.1f}-{max(areas):.1f}, "
                f"mean circularity: {np.mean(circs):.3f}"
            )

        spatial = cv_results.get("spatial_features", {})
        spatial_summary = json.dumps(spatial, indent=2) if spatial else "N/A"

        return {
            "system_prompt": (
                "You are a forensic bloodstain pattern analysis expert. "
                "Synthesize the computer vision analysis, AI visual analysis, "
                "and experimental metadata into a comprehensive assessment."
            ),
            "user_prompt": (
                f"AI Visual Analysis:\n{json.dumps(vision_results, indent=2)}\n\n"
                f"Computer Vision Features:\n{stain_summary}\n"
                f"Spatial: {spatial_summary}\n\n"
                f"Experiment Metadata:\n{json.dumps(metadata or {}, indent=2)}\n\n"
                f"Provide a combined analysis with reasoning about the pattern, "
                f"mechanism, and any relationships between the evidence elements."
            ),
        }

    def build_extraction_result(
        self,
        analysis: Dict,
        cv_features: Dict,
        metadata: Optional[Dict],
    ) -> Dict[str, Any]:
        """Convert analysis into entities + relationships format."""
        exp_id = (metadata or {}).get("experiment_id", "unknown")
        entities = []
        relationships = []

        # Prefix all internal IDs with exp_id to avoid collisions across experiments
        pattern_id = f"{exp_id}_pattern"
        entities.append({
            "entity_type": "BloodstainPattern",
            "entity_id": pattern_id,
            "properties": {
                "pattern_id": pattern_id,
                "pattern_type": analysis.get("pattern_type", "unknown"),
                "confidence": analysis.get("confidence", 0.5),
                "description": analysis.get("description", ""),
                "key_features": json.dumps(analysis.get("key_features", [])),
                "uncertainties": json.dumps(analysis.get("uncertainties", [])),
                "spatial_distribution": analysis.get("spatial_distribution", ""),
                "estimated_stain_count": analysis.get(
                    "estimated_stain_count",
                    cv_features.get("total_stains", 0),
                ),
            },
        })

        for sf in cv_features.get("stain_features", []):
            raw_stain_id = sf.get("stain_id", f"stain_{len(entities)}")
            stain_id = f"{exp_id}_{raw_stain_id}"
            stain_props = {k: v for k, v in sf.items() if k != "stain_id"}
            entities.append({
                "entity_type": "Stain",
                "entity_id": stain_id,
                "properties": {"stain_id": stain_id, **stain_props},
            })
            relationships.append({
                "source_entity_id": pattern_id,
                "target_entity_id": stain_id,
                "relationship_type": "CONTAINS_STAIN",
                "confidence": 0.95,
                "evidence_span": "CV segmentation",
            })

        mechanism_id = f"{exp_id}_mechanism"
        entities.append({
            "entity_type": "ImpactMechanismNode",
            "entity_id": mechanism_id,
            "properties": {
                "mechanism_id": mechanism_id,
                "mechanism_type": analysis.get("impact_mechanism", "unknown"),
                "confidence": analysis.get("confidence", 0.5),
                "description": analysis.get("reasoning", ""),
            },
        })
        relationships.append({
            "source_entity_id": pattern_id,
            "target_entity_id": mechanism_id,
            "relationship_type": "GENERATED_BY",
            "confidence": analysis.get("confidence", 0.5),
            "evidence_span": "GPT-4o vision analysis",
        })

        if metadata and metadata.get("experiment_id"):
            entities.append({
                "entity_type": "Experiment",
                "entity_id": exp_id,
                "properties": {k: v for k, v in metadata.items() if v is not None},
            })
            relationships.append({
                "source_entity_id": pattern_id,
                "target_entity_id": exp_id,
                "relationship_type": "CAUSED_BY",
                "confidence": 1.0,
                "evidence_span": "Experiment metadata",
            })

        return {
            "entities": entities,
            "relationships": relationships,
            "metadata": {
                "cv_stains_detected": cv_features.get("total_stains", 0),
                "pattern_type": analysis.get("pattern_type"),
                "mechanism": analysis.get("impact_mechanism"),
                "confidence": analysis.get("confidence"),
            },
        }

    def _extract_geometric_features(self, contour: np.ndarray) -> Dict[str, Any]:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h if h > 0 else 0
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        circularity = (
            4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
        )

        orientation = 0.0
        eccentricity = 0.0
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                orientation = float(ellipse[2])
                a_ax = max(ellipse[1]) / 2
                b_ax = min(ellipse[1]) / 2
                if a_ax > 0:
                    ratio = min((b_ax / a_ax) ** 2, 1.0)
                    eccentricity = float(np.sqrt(1 - ratio))
            except cv2.error:
                pass

        return {
            "area": float(area),
            "perimeter": float(perimeter),
            "aspect_ratio": float(aspect_ratio),
            "circularity": float(circularity),
            "solidity": float(solidity),
            "orientation": orientation,
            "eccentricity": eccentricity,
        }

    def _extract_spatial_features(
        self, stains: List[Dict], image_shape: tuple
    ) -> Dict[str, Any]:
        if not stains:
            return {}

        centers = [
            (s.get("center_x", 0), s.get("center_y", 0))
            for s in stains
            if s.get("center_x") and s.get("center_y")
        ]
        if len(centers) < 2:
            return {"stain_count": len(stains)}

        centers_arr = np.array(centers)
        from scipy.spatial.distance import cdist

        dists = cdist(centers_arr, centers_arr)
        np.fill_diagonal(dists, np.inf)
        nn_dists = dists.min(axis=1)

        return {
            "stain_count": len(stains),
            "mean_nearest_neighbor": float(np.mean(nn_dists)),
            "std_nearest_neighbor": float(np.std(nn_dists)),
            "spatial_extent_x": float(centers_arr[:, 0].max() - centers_arr[:, 0].min()),
            "spatial_extent_y": float(centers_arr[:, 1].max() - centers_arr[:, 1].min()),
            "centroid_x": float(np.mean(centers_arr[:, 0])),
            "centroid_y": float(np.mean(centers_arr[:, 1])),
        }
