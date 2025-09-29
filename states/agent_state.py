"""
State definitions for the market research workflow
"""
from typing import Dict, Any
from pydantic import BaseModel

class MarketResearchState(BaseModel):
    """State schema for the market research workflow"""
    # Configuration
    groq_api_key: str
    
    # Input
    product_description: str
    
    # Market Analysis
    optimistic_analysis: str | None = None
    pessimistic_analysis: str | None = None
    market_synthesis: str | None = None
    
    # Innovation
    current_proposal: str | None = None
    current_critique: str | None = None
    final_innovation: str | None = None
    iteration_count: int = 0
    
    # Output
    pitch_deck: str | None = None

    def dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None values"""
        return {k: v for k, v in super().dict().items() if v is not None}