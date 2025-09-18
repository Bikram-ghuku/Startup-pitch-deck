from typing import TypedDict, Annotated, List, Dict, Optional
from enum import Enum
from dataclasses import dataclass
from langchain_core.messages import BaseMessage
import time

class AgentStage(Enum):
    INITIAL = "initial"
    MARKET_ANALYSIS = "market_analysis"
    INNOVATION = "innovation"
    DEBATE = "debate"
    COMPLETE = "complete"

class MarketPerspective(Enum):
    OPTIMISTIC = "optimistic"
    PESSIMISTIC = "pessimistic"
    SYNTHESIS = "synthesis"

@dataclass
class SearchEvent:
    query: str
    results: List[str]
    timestamp: float = time.time()

@dataclass
class MarketAnalysis:
    perspective: MarketPerspective
    analysis: str
    evidence: List[str]
    confidence: float  # 0-1 scale
    timestamp: float = time.time()

@dataclass
class MarketDebateRound:
    optimistic_view: MarketAnalysis
    pessimistic_view: MarketAnalysis
    synthesis: str
    key_findings: List[str]
    timestamp: float = time.time()

@dataclass
class DebateRound:
    innovation_point: str
    debate_critique: str
    conclusion: str
    timestamp: float = time.time()

def add_search_events(current: List[SearchEvent], new: List[SearchEvent]) -> List[SearchEvent]:
    """Reducer for combining search event lists."""
    return current + new

def add_market_analyses(current: Dict[MarketPerspective, MarketAnalysis],
                       new: Dict[MarketPerspective, MarketAnalysis]) -> Dict[MarketPerspective, MarketAnalysis]:
    """Reducer for combining market analyses."""
    return {**current, **new}

def add_debate_rounds(current: List[MarketDebateRound], new: List[MarketDebateRound]) -> List[MarketDebateRound]:
    """Reducer for combining debate rounds."""
    return current + new

def add_innovation_rounds(current: List[DebateRound], new: List[DebateRound]) -> List[DebateRound]:
    """Reducer for combining innovation debate rounds."""
    return current + new

def append_insights(current: List[str], new: List[str]) -> List[str]:
    """Reducer for combining market insights."""
    return current + new

class AgentState(TypedDict):
    """LangGraph-compatible state management for market research agents."""
    
    # Core context and messages
    context: str
    stage: AgentStage
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]
    
    # Search and analysis tracking
    search_history: Annotated[List[SearchEvent], add_search_events]
    current_market_analysis: Annotated[Dict[MarketPerspective, MarketAnalysis], add_market_analyses]
    
    # Debate histories (internal only)
    market_debate_history: Annotated[List[MarketDebateRound], add_debate_rounds]
    innovation_debate_history: Annotated[List[DebateRound], add_innovation_rounds]
    
    # Final outputs
    final_market_insights: Annotated[List[str], append_insights]
    final_innovations: Annotated[List[str], append_insights]
    final_risks: Annotated[List[str], append_insights]
    final_ratings: Dict[str, float]
    market_opportunity_score: float

def create_initial_state(context: str) -> AgentState:
    """Create a new agent state instance."""
    return AgentState(
        context=context,
        stage=AgentStage.INITIAL,
        messages=[],
        search_history=[],
        current_market_analysis={},
        market_debate_history=[],
        innovation_debate_history=[],
        final_market_insights=[],
        final_innovations=[],
        final_risks=[],
        final_ratings={},
        market_opportunity_score=0.0
    )

# Helper functions for state updates
def add_message(state: AgentState, role: str, content: str) -> AgentState:
    """Add a message to the conversation history."""
    return {
        **state,
        "messages": state["messages"] + [BaseMessage(role=role, content=content)]
    }

def add_search(state: AgentState, query: str, results: List[str]) -> AgentState:
    """Add a search event to history."""
    return {
        **state,
        "search_history": state["search_history"] + [SearchEvent(query=query, results=results)]
    }

def update_stage(state: AgentState, new_stage: AgentStage) -> AgentState:
    """Update the current stage."""
    new_state = {
        **state,
        "stage": new_stage
    }
    return add_message(new_state, "system", f"Moving to stage: {new_stage.value}")

def add_market_analysis(state: AgentState, 
                       perspective: MarketPerspective,
                       analysis: str,
                       evidence: List[str],
                       confidence: float) -> AgentState:
    """Add a market analysis to current state."""
    new_analysis = MarketAnalysis(
        perspective=perspective,
        analysis=analysis,
        evidence=evidence,
        confidence=confidence
    )
    return {
        **state,
        "current_market_analysis": {
            **state["current_market_analysis"],
            perspective: new_analysis
        }
    }

def add_market_debate_round(state: AgentState,
                           synthesis: str,
                           key_findings: List[str]) -> AgentState:
    """Add a market debate round to history."""
    if (MarketPerspective.OPTIMISTIC in state["current_market_analysis"] and
        MarketPerspective.PESSIMISTIC in state["current_market_analysis"]):
        new_round = MarketDebateRound(
            optimistic_view=state["current_market_analysis"][MarketPerspective.OPTIMISTIC],
            pessimistic_view=state["current_market_analysis"][MarketPerspective.PESSIMISTIC],
            synthesis=synthesis,
            key_findings=key_findings
        )
        new_state = {
            **state,
            "market_debate_history": state["market_debate_history"] + [new_round],
            "current_market_analysis": {},  # Clear current analysis
            "final_market_insights": state["final_market_insights"] + key_findings
        }
        return new_state
    return state

def add_innovation_debate(state: AgentState,
                         innovation: str,
                         critique: str,
                         conclusion: str) -> AgentState:
    """Add an innovation debate round to history."""
    new_round = DebateRound(
        innovation_point=innovation,
        debate_critique=critique,
        conclusion=conclusion
    )
    return {
        **state,
        "innovation_debate_history": state["innovation_debate_history"] + [new_round]
    }