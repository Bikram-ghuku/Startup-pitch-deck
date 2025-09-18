from typing import List
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from states.agent_state import (
    AgentState, MarketPerspective, AgentStage,
    add_search, add_market_analysis, update_stage
)
import json

load_dotenv()

_optimist_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.8,  # Higher temperature for more enthusiastic responses
    max_tokens=1024
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
        
        f"CONTEXT:\n{state['context']}\n\n"
        
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
        
        "TOOL USAGE:\n"
        "When you need to search for information, use the following format:\n"
        "SEARCH: your specific search query\n\n"
        "Example searches:\n"
        "SEARCH: market size smart home automation 2025\n"
        "SEARCH: emerging trends in personalization technology\n\n"
        
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
    
    # Store the response in state for verbose output
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
    # Higher confidence if we got actual results vs "No results" or errors
    evidence_quality = sum(1 for e in evidence if "No results" not in e and "Search unavailable" not in e)
    confidence = min(0.5 + (evidence_quality * 0.1), 1.0)
    
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
