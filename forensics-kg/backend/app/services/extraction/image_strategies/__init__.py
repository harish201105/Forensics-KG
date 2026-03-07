"""Image analysis strategies for different forensic image types."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class ImageAnalysisStrategy(ABC):
    """Abstract base class for forensic image analysis strategies.

    Each strategy implements a 3-stage pipeline:
    1. preprocess: OpenCV-based traditional computer vision
    2. vision_analysis: GPT-4o vision API analysis
    3. combine: Synthesize CV + Vision into final entities/relationships
    """

    @abstractmethod
    async def preprocess(self, image_bytes: bytes) -> Dict[str, Any]:
        """OpenCV preprocessing, returns CV metrics/features."""

    @abstractmethod
    async def vision_analysis(
        self, image_b64: str, cv_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """GPT-4o Vision analysis with CV context."""

    @abstractmethod
    async def combine(
        self,
        cv_results: Dict[str, Any],
        vision_results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Synthesize CV + Vision into final entities/relationships."""

    @abstractmethod
    def get_schema(self) -> dict:
        """Return the JSON schema for structured vision output."""

    @abstractmethod
    def get_combined_schema(self) -> dict:
        """Return the JSON schema for combined analysis output."""

    @abstractmethod
    def build_extraction_result(
        self,
        analysis: Dict[str, Any],
        cv_features: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Build final entities + relationships from analysis results."""


# Strategy registry — populated by imports below
STRATEGY_REGISTRY: Dict[str, ImageAnalysisStrategy] = {}


def register_strategy(name: str, strategy: ImageAnalysisStrategy) -> None:
    """Register an image analysis strategy."""
    STRATEGY_REGISTRY[name] = strategy


def get_strategy(image_type: str) -> ImageAnalysisStrategy:
    """Get the strategy for a given image type."""
    strategy = STRATEGY_REGISTRY.get(image_type)
    if strategy is None:
        raise ValueError(
            f"Unknown image type '{image_type}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return strategy
