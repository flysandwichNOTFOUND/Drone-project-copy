"""DeepSeek API boundary; request schema and model are pending."""

from drone_sim.config import DeepSeekConfig
from drone_sim.models import HazardAssessment, InspectionEvidence


class DeepSeekAnalyzer:
    def __init__(self, config: DeepSeekConfig):
        self.config = config

    async def assess(self, evidence: InspectionEvidence) -> HazardAssessment:
        """Assess structured observations with a future async API call."""
        raise NotImplementedError("DeepSeek API integration is pending.")

    async def close(self) -> None:
        """Release the API session when one is implemented."""
