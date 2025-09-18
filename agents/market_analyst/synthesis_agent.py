from typing import List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from states.agent_state import AgentState, MarketPerspective, AgentStage
import json

load_dotenv()

_synthesis_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.4,  # Balanced temperature for objective analysis
    max_tokens=4096,
)

def synthesis_agent(state: AgentState) -> AgentState:
    """Synthesizes optimistic and pessimistic market analyses into balanced insights."""
    
    # Get relevant state information
    optimist_view = state["current_market_analysis"].get(MarketPerspective.OPTIMISTIC)
    pessimist_view = state["current_market_analysis"].get(MarketPerspective.PESSIMISTIC)
    
    if not (optimist_view and pessimist_view):
        raise ValueError("Both optimistic and pessimistic analyses required for synthesis")
    
    debate_history = "\n".join([
        f"Round {i+1}:\n{round.synthesis}\nKey Findings: {', '.join(round.key_findings)}"
        for i, round in enumerate(state["market_debate_history"][-2:])
    ]) if state["market_debate_history"] else ""
    
    current_risks = "\n".join(state["final_risks"]) if state["final_risks"] else "None identified"
    final_insights = "\n".join(state["final_market_insights"]) if state["final_market_insights"] else "None yet"
    
    prompt = (
        "You are an objective market analysis synthesizer. Your role is to evaluate "
        "competing market perspectives and create a balanced, evidence-based synthesis. "
        "Consider both optimistic opportunities and realistic challenges.\n\n"
        
        f"CONTEXT:\n{context}\n\n"
        
        "PERSPECTIVES TO SYNTHESIZE:\n"
        f"Optimistic View:\n{optimist_analysis}\n\n"
        f"Pessimistic View:\n{pessimist_analysis}\n\n"
        
        "TOOLS AVAILABLE:\n"
        "- web_search: Verify claims and gather additional data\n"
        "- market_data: Validate market statistics\n"
        "- competitor_analysis: Verify competitive landscape\n\n"
        
        "SYNTHESIS FRAMEWORK:\n"
        "1. Evidence Evaluation\n"
        "- Validate key claims\n"
        "- Cross-reference data\n"
        "- Identify common ground\n\n"
        
        "2. Market Reality Assessment\n"
        "- Balanced opportunity analysis\n"
        "- Realistic risk evaluation\n"
        "- Market timing considerations\n\n"
        
        "3. Strategic Implications\n"
        "- Key success factors\n"
        "- Critical challenges\n"
        "- Strategic recommendations\n\n"
        
        "4. Supporting Research\n"
        "- Additional data needs\n"
        "- Verification points\n"
        "- Expert perspectives\n\n"
        
        "FORMAT:\n"
        "Present your synthesis as a series of tool calls and insights. "
        "Use the following structure for tool calls:\n"
        "```json\n"
        '{"tool": "tool_name", "parameters": {"param1": "value1"}}\n'
        "```\n\n"
        
        "DELIVERABLES:\n"
        "1. Synthesis Summary\n"
        "- Key agreements between perspectives\n"
        "- Critical differences\n"
        "- Supporting evidence\n\n"
        
        "2. Market Assessment\n"
        "- Opportunity sizing\n"
        "- Risk quantification\n"
        "- Success probability\n\n"
        
        "3. Recommendations\n"
        "- Strategic approach\n"
        "- Risk mitigation\n"
        "- Success metrics\n\n"
        
        "IMPORTANT:\n"
        "- Maintain objectivity\n"
        "- Focus on evidence\n"
        "- Acknowledge uncertainties\n"
        "- Provide actionable insights\n"
        "- Consider timing factors\n\n"
        
        "Begin your synthesis of the market perspectives."
    )
    
    # Get LLM response
    response = _synthesis_llm.invoke(prompt)
    response_text = getattr(response, "content", str(response))
    
    # Extract tool calls and insights
    tool_calls = []
    insights = []
    recommendations = []
    
    for line in response_text.split("\n"):
        if line.strip().startswith("{") and line.strip().endswith("}"):
            try:
                tool_call = json.loads(line.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
        elif line.strip().startswith("RECOMMENDATION:"):
            recommendations.append(line.strip()[15:].strip())
        elif line.strip():
            insights.append(line.strip())
    
    # Process tool calls and gather evidence
    evidence = []
    for tool_call in tool_calls:
        if tool_call.get("tool") == "web_search":
            query = tool_call.get("parameters", {}).get("query", "")
            if query:
                state = add_search(state, query, ["Search executed"])
                evidence.append(f"Search: {query}")
    
    # Add synthesis to state and record debate round
    state = add_market_debate_round(
        state=state,
        synthesis=response_text,
        key_findings=recommendations
    )
    
    # Update final insights and market score
    if recommendations:
        # Calculate market opportunity score based on both perspectives and synthesis
        optimist_confidence = optimist_view.confidence
        pessimist_confidence = pessimist_view.confidence
        evidence_strength = len(evidence) * 0.05
        
        # Weighted score favoring the more confident perspective but tempered by evidence
        market_score = (
            (optimist_confidence * 0.6) +  # Optimist view weighted more
            (pessimist_confidence * 0.4) +  # Pessimist view weighted less
            evidence_strength  # Bonus for evidence
        ) / (1 + evidence_strength)  # Normalize to 0-1
        
        market_score = min(market_score, 1.0)
        
        # Update state with new score
        state = {
            **state,
            "market_opportunity_score": market_score
        }
    
    # Move to next stage
    if state["stage"] == AgentStage.MARKET_ANALYSIS:
        state = update_stage(state, AgentStage.INNOVATION)
    
    return state
