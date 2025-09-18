from typing import List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from states.agent_state import AgentState, MarketPerspective
import json

load_dotenv()

_optimist_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.8,  # Higher temperature for more enthusiastic responses
    max_tokens=4096,
)

def optimistic_analyst(state: AgentState) -> AgentState:
    """Optimistic market analyst that seeks growth opportunities and positive market signals."""
    
    # Get relevant state information
    debate_history = "\n".join([
        f"Round {i+1}:\n{round.synthesis}\nKey Findings: {', '.join(round.key_findings)}"
        for i, round in enumerate(state["market_debate_history"][-2:])
    ]) if state["market_debate_history"] else ""
    
    current_analysis = state["current_market_analysis"].get(MarketPerspective.OPTIMISTIC)
    final_insights = "\n".join(state["final_market_insights"]) if state["final_market_insights"] else "None yet"
    
    prompt = (
        "You are an enthusiastic market analyst with a strong growth mindset. "
        "Your role is to identify and validate market opportunities, focusing on positive signals "
        "and growth potential.\n\n"
        
        f"CONTEXT:\n{context}\n\n"
        
        "TOOLS AVAILABLE:\n"
        "- web_search: Research market trends and opportunities\n"
        "- market_data: Access market size and growth data\n"
        "- competitor_analysis: Analyze competitor landscape\n\n"
        
        "APPROACH:\n"
        "1. Market Opportunity Analysis\n"
        "- Identify growing market segments\n"
        "- Highlight positive market signals\n"
        "- Find success stories and analogies\n\n"
        
        "2. Growth Potential\n"
        "- Market expansion possibilities\n"
        "- User adoption drivers\n"
        "- Revenue potential\n\n"
        
        "3. Competitive Advantage\n"
        "- Unique value propositions\n"
        "- Market entry timing\n"
        "- Strategic positioning\n\n"
        
        "4. Supporting Evidence\n"
        "- Market research data\n"
        "- Industry trends\n"
        "- Expert opinions\n\n"
        
        "FORMAT:\n"
        "Present your analysis as a series of tool calls and insights. "
        "Use the following structure for tool calls:\n"
        "```json\n"
        '{"tool": "tool_name", "parameters": {"param1": "value1"}}\n'
        "```\n\n"
        
        "IMPORTANT:\n"
        "- Focus on opportunities and growth\n"
        "- Back claims with data\n"
        "- Be enthusiastic but factual\n"
        "- Build on previous insights\n"
        "- Maintain professional optimism\n\n"
        
        "Begin your market opportunity analysis."
    )
    
    # Get LLM response
    response = _optimist_llm.invoke(prompt)
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
    confidence = min(0.5 + (len(evidence) * 0.1), 1.0)
    
    # Add analysis to state
    state = add_market_analysis(
        state=state,
        perspective=MarketPerspective.OPTIMISTIC,
        analysis=response_text,
        evidence=evidence,
        confidence=confidence
    )
    
    # Update state stage if needed
    if state["stage"] == AgentStage.INITIAL:
        state = update_stage(state, AgentStage.MARKET_ANALYSIS)
    
    return state
