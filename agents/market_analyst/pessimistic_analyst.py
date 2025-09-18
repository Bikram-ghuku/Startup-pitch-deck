from typing import List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from states.agent_state import AgentState, MarketPerspective
import json

load_dotenv()

_pessimist_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,  # Lower temperature for more conservative analysis
    max_tokens=4096,
)

def pessimistic_analyst(state: AgentState) -> AgentState:
    """Pessimistic market analyst that identifies risks and market challenges."""
    
    # Get relevant state information
    debate_history = "\n".join([
        f"Round {i+1}:\n{round.synthesis}\nKey Findings: {', '.join(round.key_findings)}"
        for i, round in enumerate(state["market_debate_history"][-2:])
    ]) if state["market_debate_history"] else ""
    
    optimist_view = state["current_market_analysis"].get(MarketPerspective.OPTIMISTIC)
    if not optimist_view:
        raise ValueError("Optimistic analysis required before pessimistic review")
    
    current_risks = "\n".join(state["final_risks"]) if state["final_risks"] else "None identified"
    final_insights = "\n".join(state["final_market_insights"]) if state["final_market_insights"] else "None yet"
    
    prompt = (
        "You are a cautious market analyst with a strong focus on risk assessment. "
        "Your role is to identify potential market challenges, barriers to entry, and "
        "competitive threats. Challenge optimistic assumptions with market realities.\n\n"
        
        f"CONTEXT:\n{context}\n\n"
        
        f"OPTIMISTIC VIEW TO CHALLENGE:\n{optimist_view}\n\n"
        
        "TOOLS AVAILABLE:\n"
        "- web_search: Research market challenges and failures\n"
        "- market_data: Access market size and decline data\n"
        "- competitor_analysis: Analyze competitive threats\n\n"
        
        "APPROACH:\n"
        "1. Market Risk Analysis\n"
        "- Identify market barriers\n"
        "- Highlight negative trends\n"
        "- Find failure cases and lessons\n\n"
        
        "2. Competitive Threats\n"
        "- Incumbent advantages\n"
        "- Market saturation\n"
        "- Entry barriers\n\n"
        
        "3. Adoption Challenges\n"
        "- User resistance factors\n"
        "- Implementation hurdles\n"
        "- Cost barriers\n\n"
        
        "4. Supporting Evidence\n"
        "- Market failure data\n"
        "- Industry challenges\n"
        "- Expert warnings\n\n"
        
        "FORMAT:\n"
        "Present your analysis as a series of tool calls and insights. "
        "Use the following structure for tool calls:\n"
        "```json\n"
        '{"tool": "tool_name", "parameters": {"param1": "value1"}}\n'
        "```\n\n"
        
        "IMPORTANT:\n"
        "- Focus on realistic challenges\n"
        "- Back concerns with data\n"
        "- Be thorough but fair\n"
        "- Challenge optimistic assumptions\n"
        "- Maintain professional skepticism\n\n"
        
        "Begin your risk analysis."
    )
    
    # Get LLM response
    response = _pessimist_llm.invoke(prompt)
    response_text = getattr(response, "content", str(response))
    
    # Extract tool calls and insights
    tool_calls = []
    insights = []
    
    for line in response_text.split("\n"):
        if line.strip().startswith("{") and line.strip().endswith("}"):
            try:
                tool_call = json.loads(line.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError:
                continue
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
    
    # Calculate confidence based on evidence strength and tool call success
    confidence = min(0.4 + (len(evidence) * 0.1), 0.9)  # Slightly lower confidence ceiling for pessimist
    
    # Add analysis to state
    state = add_market_analysis(
        state=state,
        perspective=MarketPerspective.PESSIMISTIC,
        analysis=response_text,
        evidence=evidence,
        confidence=confidence
    )
    
    # Update state stage if needed
    if state["stage"] == AgentStage.INITIAL:
        state = update_stage(state, AgentStage.MARKET_ANALYSIS)
    
    return state