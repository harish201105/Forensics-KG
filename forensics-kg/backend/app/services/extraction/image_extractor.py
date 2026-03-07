"""Image analysis orchestrator using strategy pattern for multiple forensic image types."""

import base64
import cv2
import numpy as np
from typing import Dict, Any, Optional
from loguru import logger

from app.services.extraction.openai_client import OpenAIClient
from app.services.extraction.image_strategies import (
    ImageAnalysisStrategy,
    register_strategy,
    get_strategy,
)
from app.services.extraction.image_strategies.bloodstain import BloodstainStrategy
from app.services.extraction.image_strategies.fingerprint import FingerprintStrategy
from app.services.extraction.image_strategies.wound import WoundStrategy
from app.services.extraction.image_strategies.ballistics import BallisticsStrategy
from app.services.extraction.image_strategies.document import DocumentForensicsStrategy
from app.services.extraction.image_strategies.toolmarks import ToolMarksStrategy


class ImageExtractor:
    """Forensic image analysis orchestrator.

    Delegates to type-specific strategies via the strategy pattern.
    Each strategy implements: preprocess (CV) -> vision_analysis (GPT-4o) -> combine.
    """

    def __init__(self, openai_client: OpenAIClient, settings):
        self._openai = openai_client
        self._settings = settings

        # Register all strategies
        self._register_strategies()

    def _register_strategies(self):
        """Register all available image analysis strategies."""
        register_strategy("bloodstain", BloodstainStrategy(self._settings))
        register_strategy("fingerprint", FingerprintStrategy(self._settings))
        register_strategy("wound", WoundStrategy(self._settings))
        register_strategy("ballistics", BallisticsStrategy(self._settings))
        register_strategy("document_forensics", DocumentForensicsStrategy(self._settings))
        register_strategy("tool_marks", ToolMarksStrategy(self._settings))

    async def analyze(
        self,
        image_bytes: bytes,
        experiment_metadata: Optional[Dict[str, Any]] = None,
        image_type: str = "bloodstain",
    ) -> Dict[str, Any]:
        """
        Analyze a forensic image using the appropriate strategy.

        Args:
            image_bytes: Raw image bytes
            experiment_metadata: Optional metadata dict
            image_type: Type of forensic image (bloodstain, fingerprint, wound,
                       ballistics, document_forensics, tool_marks)
        """
        strategy = get_strategy(image_type)
        logger.info(f"Analyzing {image_type} image with {strategy.__class__.__name__}")

        # Step 1: CV preprocessing
        cv_features = await strategy.preprocess(image_bytes)
        logger.info(f"CV preprocessing complete for {image_type}")

        # Step 2: Resize image for GPT-4o (keep under 20MB)
        resized_bytes = self._resize_for_api(image_bytes)
        image_b64 = base64.b64encode(resized_bytes).decode("utf-8")

        # Step 3: GPT-4o vision analysis
        vision_prompts = await strategy.vision_analysis(image_b64, cv_features)
        vision_analysis = await self._openai.analyze_image(
            system_prompt=vision_prompts["system_prompt"],
            user_prompt=vision_prompts["user_prompt"],
            image_base64=image_b64,
            response_schema=strategy.get_schema(),
        )
        logger.info(f"Vision analysis complete for {image_type}")

        # Step 4: Combined analysis
        combined_prompts = await strategy.combine(cv_features, vision_analysis, experiment_metadata)
        if cv_features and (experiment_metadata or self._has_meaningful_cv(cv_features)):
            combined = await self._openai.extract_structured(
                system_prompt=combined_prompts["system_prompt"],
                user_prompt=combined_prompts["user_prompt"],
                response_schema=strategy.get_combined_schema(),
            )
        else:
            combined = vision_analysis

        # Step 5: Build entities + relationships
        return strategy.build_extraction_result(combined, cv_features, experiment_metadata or {})

    def _has_meaningful_cv(self, cv_features: Dict) -> bool:
        """Check if CV preprocessing produced meaningful results."""
        # Bloodstain
        if cv_features.get("total_stains", 0) > 0:
            return True
        # Fingerprint
        if cv_features.get("minutiae_count", 0) > 0:
            return True
        # Wound
        if cv_features.get("wound_detected", False):
            return True
        # Ballistics/tool marks
        if cv_features.get("striation_count", 0) > 0:
            return True
        # Document
        if cv_features.get("text_detected", False):
            return True
        return False

    def _resize_for_api(self, image_bytes: bytes, max_dim: int = 2048) -> bytes:
        """Resize image for API submission."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return image_bytes
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes()

    @staticmethod
    def get_supported_image_types():
        """Return list of supported image types."""
        return [
            "bloodstain", "fingerprint", "wound",
            "ballistics", "document_forensics", "tool_marks",
        ]
