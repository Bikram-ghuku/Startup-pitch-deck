from typing import List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from states.agent_state import (
    AgentState, MarketPerspective, AgentStage,
    add_search, add_market_analysis, update_stage
)
import json

load_dotenv()

_pessimist_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.3,  # Lower temperature for more conservative analysis
    max_tokens=1024
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
        "You are a pesimmistic market analyst with a strong focus on why the idea will fail. "
        "Your role is to identify potential market challenges, barriers to entry, failure reasons and "
        "competitive threats. Challenge optimistic assumptions with market realities.\n\n"
        
        f"CONTEXT:\n{state['context']}\n\n"
        
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
        
        "TOOL USAGE:\n"
        "When you need to search for information, use the following format:\n"
        "SEARCH: your specific search query\n\n"
        "Example searches:\n"
        "SEARCH: failure cases smart home automation\n"
        "SEARCH: privacy concerns AI assistants\n\n"
        
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

    state = {
        **state,
        "current_response": response_text
    }
    
    # Process search requests and gather evidence
    evidence = []
    for line in response_text.split('\n'):
        if line.strip().startswith('SEARCH:'):
            query = line[7:].strip()
            # Call web_search tool through state management
            state = add_search(state, query)
            evidence.append(f"Search: {query} -> {state['search_results'][-1] if state['search_results'] else 'No results'}")
    
    # Calculate confidence based on evidence quality
    # Higher confidence for pessimist when finding actual challenges/risks
    evidence_quality = sum(1 for e in evidence if "No results" not in e and "Search unavailable" not in e)
    # Pessimist gets higher confidence from finding problems
    confidence = min(0.4 + (evidence_quality * 0.15), 0.9)  # Can reach 0.9 with good evidence
    
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