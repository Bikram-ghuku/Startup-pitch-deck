"""
LangGraph implementation for market research workflow.
Orchestrates the flow between market analysis, innovation, and pitch deck agents.
"""
from typing import Dict, Any
from langgraph.graph import StateGraph
from states.agent_state import MarketResearchState

from agents.market_analyst.optimistic_analyst import analyze_optimistically
from agents.market_analyst.pessimistic_analyst import analyze_pessimistically
from agents.market_analyst.synthesis_agent import synthesize_analysis
from agents.innovation.innovation_agent import generate_innovation_proposal
from agents.innovation.debater_agent import debate_proposal
from agents.pitch_deck.pitch_deck import generate_pitch_deck

from IPython.display import Image, display

async def market_research_graph(
    product_description: str,
    groq_api_key: str,
    max_iterations: int = 3
) -> Dict[str, Any]:
    """
    Create and execute the market research workflow graph.
    
    Args:
        product_description: Description of the product to analyze
        groq_api_key: API key for ChatGroq
        max_iterations: Maximum number of innovation-debate iterations
    """
    
    # Create the graph with state schema
    workflow = StateGraph(MarketResearchState)

    # Add market analysis nodes
    workflow.add_node("optimistic_analysis", analyze_optimistically)
    workflow.add_node("pessimistic_analysis", analyze_pessimistically)
    workflow.add_node("synthesis", synthesize_analysis)
    
    # Add innovation nodes
    workflow.add_node("innovation", generate_innovation_proposal)
    workflow.add_node("debate", debate_proposal)
    
    # Add conditional node for iteration control
    def should_continue_iteration(state: MarketResearchState):
        if state.iteration_count >= max_iterations:
            return {"next": "create_pitch_deck"}
        return {"next": "innovation"}
    
    workflow.add_node("check_iteration", should_continue_iteration)
    
    # Add pitch deck node
    workflow.add_node("create_pitch_deck", generate_pitch_deck)

    # Define the edges
    # Market analysis flow
    workflow.add_edge("optimistic_analysis", "pessimistic_analysis")
    workflow.add_edge("pessimistic_analysis", "synthesis")
    workflow.add_edge("synthesis", "innovation")
    
    # Innovation-debate loop
    workflow.add_edge("innovation", "debate")
    workflow.add_edge("debate", "check_iteration")
    
    # Conditional branching
    workflow.add_conditional_edges(
        "check_iteration",
        should_continue_iteration,
        lambda x: x["next"]
    )

    # Set the entry point
    workflow.set_entry_point("optimistic_analysis")

    # Compile the graph
    app = workflow.compile()
    # display(Image(app.get_graph().draw_mermaid_png()))

    # Initialize the state
    initial_state = MarketResearchState(
        product_description=product_description,
        groq_api_key=groq_api_key,
        iteration_count=0
    ).dict()

    # Run the graph
    result = await app.ainvoke(initial_state)
    return result