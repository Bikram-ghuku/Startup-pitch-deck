from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END, START
from states.agent_state import AgentState, AgentStage, create_initial_state

# Import agents
from agents.market_analyst.optimistic_analyst import optimistic_analyst
from agents.market_analyst.pessimistic_analyst import pessimistic_analyst
from agents.market_analyst.synthesis_agent import synthesis_agent
from agents.innovation.innovation_agent import innovation_strategist
from agents.innovation.debater_agent import debater_agent

def should_continue_market_analysis(state: AgentState) -> str:
    """Determine if we should continue market analysis or move to innovation."""
    # Get current market score, defaulting to 0 if not set
    market_score = state.get("market_opportunity_score", 0.0)
    
    # Continue if we haven't completed at least one full round of analysis
    # (optimistic -> pessimistic -> synthesis)
    if len(state["market_debate_history"]) < 1:
        return "continue_market"
    
    # If market score is too low after analysis, end the process
    if market_score < 0.3 and len(state["market_debate_history"]) >= 1:
        return "end"
    
    # Move to innovation if we have good market understanding
    return "move_to_innovation"

def should_continue_innovation(state: AgentState) -> str:
    """Determine if we should continue innovation cycle."""
    # Continue if we haven't generated enough innovations
    if len(state["final_innovations"]) < 2:
        return "continue_innovation"
    
    # End if we've found enough good innovations
    if len(state["final_innovations"]) >= 3:
        return "end"
    
    # Continue if our last few attempts weren't successful
    recent_debates = state["innovation_debate_history"][-3:]
    successful = sum(1 for d in recent_debates if "APPROVED" in d.conclusion.upper())
    return "continue_innovation" if successful < 2 else "end"

def format_state_update(state: AgentState, stage: str, llm_response: str = None) -> str:
    """Format current state information for verbose output."""
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"Stage: {stage}")
    output.append(f"Current Stage: {state['stage'].value}")
    
    # Show LLM Response if available
    if llm_response:
        output.append("\nLLM Response:")
        output.append("-" * 40)
        # Split response into lines and indent
        response_lines = llm_response.split('\n')
        for line in response_lines:
            output.append(f"  {line}")
        output.append("-" * 40)
    
    # Market Analysis Info
    if state["market_debate_history"]:
        output.append("\nMarket Debate Progress:")
        output.append(f"- Rounds Completed: {len(state['market_debate_history'])}")
        output.append(f"- Market Score: {state['market_opportunity_score']:.2f}")
        if state["final_market_insights"]:
            output.append("- Latest Insights:")
            for insight in state["final_market_insights"][-3:]:  # Last 3 insights
                output.append(f"  * {insight}")
    
    # Innovation Progress
    if state["innovation_debate_history"]:
        output.append("\nInnovation Progress:")
        output.append(f"- Rounds Completed: {len(state['innovation_debate_history'])}")
        output.append(f"- Approved Innovations: {len(state['final_innovations'])}")
        if state["final_innovations"]:
            output.append("- Latest Innovations:")
            for innovation in state["final_innovations"][-2:]:  # Last 2 innovations
                output.append(f"  * {innovation}")
    
    # Risk Tracking
    if state["final_risks"]:
        output.append("\nRisk Tracking:")
        output.append(f"- Total Risks Identified: {len(state['final_risks'])}")
        output.append("- Latest Risks:")
        for risk in state["final_risks"][-3:]:  # Last 3 risks
            output.append(f"  * {risk}")
    
    # Search Activity
    if state["search_history"]:
        output.append("\nSearch Activity:")
        output.append(f"- Total Searches: {len(state['search_history'])}")
        output.append("- Recent Searches:")
        for search in state["search_history"][-3:]:  # Last 3 searches
            output.append(f"  * {search.query}")
            
    output.append(f"{'='*80}\n")
    return "\n".join(output)

def format_final_output(state: AgentState) -> str:
    """Format the final output for the user."""
    output = []
    
    # Market Analysis Results
    output.append("=== FINAL RESULTS ===")
    output.append("\nMarket Analysis:")
    output.append(f"- Market Opportunity Score: {state['market_opportunity_score']:.2f}")
    output.append("- Key Market Insights:")
    for insight in state["final_market_insights"]:
        output.append(f"  * {insight}")
    
    # Innovation Results
    output.append("\nApproved Innovations:")
    for i, innovation in enumerate(state["final_innovations"], 1):
        output.append(f"{i}. {innovation}")
    
    # Risks and Considerations
    output.append("\nKey Risks and Challenges:")
    for risk in state["final_risks"]:
        output.append(f"- {risk}")
    
    # Process Statistics
    output.append("\nProcess Statistics:")
    output.append(f"- Market Analysis Rounds: {len(state['market_debate_history'])}")
    output.append(f"- Innovation Rounds: {len(state['innovation_debate_history'])}")
    output.append(f"- Total Searches Performed: {len(state['search_history'])}")
    
    return "\n".join(output)

def create_verbose_node(name: str, agent_func):
    """Create a node that prints state information and LLM responses."""
    def verbose_node(state: AgentState) -> AgentState:
        print(f"\nExecuting {name}...")
        print(format_state_update(state, f"Before {name}"))
        
        new_state = agent_func(state)
        
        # Get the new response_text if available
        new_response = new_state.get("current_response", "")
        if new_response:
            print("\nLLM Response:")
            print("-" * 80)
            print(new_response)
            print("-" * 80)
        
        print(format_state_update(new_state, f"After {name}"))
        return new_state
    
    return verbose_node

def create_market_innovation_graph() -> StateGraph:
    """Create the market analysis and innovation workflow graph."""
    
    # Create the graph with our state type
    workflow = StateGraph(AgentState)
    
    # Add nodes for market analysis with verbose output
    workflow.add_node("optimistic_analysis", create_verbose_node("Optimistic Analysis", optimistic_analyst))
    workflow.add_node("pessimistic_analysis", create_verbose_node("Pessimistic Analysis", pessimistic_analyst))
    workflow.add_node("market_synthesis", create_verbose_node("Market Synthesis", synthesis_agent))
    
    # Add nodes for innovation with verbose output
    workflow.add_node("innovation_generation", create_verbose_node("Innovation Generation", innovation_strategist))
    workflow.add_node("innovation_debate", create_verbose_node("Innovation Debate", debater_agent))
    
    # Add edges for market analysis cycle
    workflow.add_edge(START, "optimistic_analysis")
    workflow.add_conditional_edges(
        "optimistic_analysis",
        should_continue_market_analysis,
        {
            "continue_market": "pessimistic_analysis",
            "move_to_innovation": "innovation_generation",
            "end": END
        }
    )
    
    workflow.add_edge("pessimistic_analysis", "market_synthesis")
    workflow.add_edge("market_synthesis", "optimistic_analysis")
    
    # Add edges for innovation cycle
    workflow.add_conditional_edges(
        "innovation_generation",
        should_continue_innovation,
        {
            "continue_innovation": "innovation_debate",
            "end": END
        }
    )
    
    workflow.add_edge("innovation_debate", "innovation_generation")
    
    # Compile the graph
    return workflow.compile()

def process_idea(idea: str) -> str:
    """Process a new idea through market analysis and innovation."""
    # Create initial state
    state = create_initial_state(idea)
    
    # Create and run the graph
    graph = create_market_innovation_graph()
    final_state = graph.invoke(state)
    
    # Format and return results
    return format_final_output(final_state)
