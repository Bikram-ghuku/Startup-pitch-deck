from langchain_groq import ChatGroq
from dotenv import load_dotenv
from tools.web_search import web_search
from states.agent_state import AgentState, AgentStage
import time

load_dotenv()

_debate_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.5,
    max_tokens=4096,
)

def debater_agent(state: AgentState) -> AgentState:
    """Debate feasibility, viability, achievability and critique reasoning process with web research.

    Args:
        context: Background, constraints, and known data points.
        proposition: The idea/feature to evaluate.
        reasoning_chain: Optional detailed reasoning from innovation agent.

    Returns:
        Structured critique of reasoning and debate outcome with evidence.
    """
    # Get relevant state information
    approved_innovations = "\n".join(state.final_innovations) if state.final_innovations else "None yet"
    current_ratings = "\n".join([f"{k}: {v}/5" for k, v in state.final_ratings.items()]) if state.final_ratings else "No ratings yet"
    current_risks = "\n".join(state.final_risks) if state.final_risks else "None identified"
    debate_rounds = len(state.debate_history)
    
    prompt = (
        f"You are a critical innovation analyst evaluating round {debate_rounds + 1}:\n\n"
        
        f"CONTEXT & HISTORY:\n"
        f"Context: {context}\n"
        f"Previously Approved: {approved_innovations}\n"
        f"Current Ratings: {current_ratings}\n"
        f"Known Risks: {current_risks}\n\n"
        
        f"PROPOSED INNOVATION:\n{proposition}\n\n"
        
        "RESEARCH TOOL:\n"
        "Use 'SEARCH: query' for validation. Examples:\n"
        "SEARCH: technical feasibility of [component]\n"
        "SEARCH: market adoption barriers for [similar solution]\n"
        "SEARCH: implementation challenges in [domain]\n\n"
        
        "EVALUATION FRAMEWORK:\n"
        "1. Historical Context Check\n"
        "- Compare with previous innovations\n"
        "- Review known risks and ratings\n"
        "- Identify unique elements\n\n"
        
        "2. Critical Analysis\n"
        "- Assumption validation\n"
        "- Evidence assessment\n"
        "- Market fit evaluation\n"
        "- Technical feasibility check\n\n"
        
        "3. Risk Assessment\n"
        "Technical Rating (0-5):\n"
        "- Implementation complexity\n"
        "- Technology readiness\n"
        "- Integration challenges\n\n"
        
        "Market Rating (0-5):\n"
        "- Market demand evidence\n"
        "- Competition analysis\n"
        "- Adoption barriers\n\n"
        
        "Resource Rating (0-5):\n"
        "- Required capabilities\n"
        "- Cost considerations\n"
        "- Timeline feasibility\n\n"
        
        "4. Risk Identification\n"
        "- New technical barriers\n"
        "- Market challenges\n"
        "- Resource constraints\n"
        "- Compare with known risks\n\n"
        
        "5. Final Verdict\n"
        "- Status: [APPROVED/REJECTED/CONDITIONAL]\n"
        "- Critical requirements\n"
        "- Improvement suggestions\n"
        "- Integration with existing innovations\n\n"
        
        "IMPORTANT:\n"
        "- Consider previous debate outcomes\n"
        "- Highlight new vs. known risks\n"
        "- Be specific about ratings\n"
        "- Provide evidence-based critique\n"
        "- Suggest actionable improvements\n\n"
        
        "Format with clear headings. Focus on novel insights and cumulative learning from previous rounds."
    )
    )
    def _handle_search(response_text: str, state: AgentState) -> str:
        """Process LLM response and handle any search requests."""
        lines = response_text.split('\n')
        result = []
        i = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('SEARCH:'):
                query = line[7:].strip()
                search_results = web_search(query)
                # Add to state
                state.add_search(query, search_results if isinstance(search_results, list) else [str(search_results)])
                result.append(f"Search Results for '{query}':")
                result.append(str(search_results))
                result.append("")
            else:
                result.append(lines[i])
            i += 1
        return '\n'.join(result)

    # Get current debate round
    current_round = state["innovation_debate_history"][-1] if state["innovation_debate_history"] else None
    if not current_round:
        raise ValueError("No innovation point found for debate")
    
    # Get debate history context
    debate_history = "\n".join([
        f"Innovation: {round.innovation_point}\n"
        f"Critique: {round.debate_critique}\n"
        f"Conclusion: {round.conclusion}\n---"
        for round in state["innovation_debate_history"][-2:]  # Last 2 rounds
    ]) if len(state["innovation_debate_history"]) > 1 else ""
    
    # Get current analysis and risks
    current_analysis = state["current_analysis"].get('latest_insight', '')
    current_risks = "\n".join(state["final_risks"]) if state["final_risks"] else "None identified"
    
    # Format prompt with state
    prompt = prompt.format(
        context=f"{state.context}{debate_context}",
        proposition=current_round.innovation_point
    )

    # Initial response
    response = _debate_llm.invoke(prompt)
    response_text = getattr(response, "content", str(response))
    
    # Handle any searches and get final response
    critique = _handle_search(response_text, state)
    
    # Extract key information from critique
    # Note: This assumes the critique follows the structured format from the prompt
    lines = critique.split('\n')
    ratings = {}
    risks = []
    conclusion = ""
    
    for line in lines:
        if ': ' in line:
            key, value = line.split(': ', 1)
            if 'Rating' in key and '/' in value:
                rating = float(value.split('/')[0])
                ratings[key.strip()] = rating
        elif line.startswith('- Risk:'):
            risks.append(line[8:].strip())
        elif line.startswith('CONCLUSION:'):
            conclusion = line[11:].strip()
    
    # Create updated debate round
    state = add_innovation_debate(
        state=state,
        innovation=current_round.innovation_point,
        critique=critique,
        conclusion=conclusion
    )
    
    # Store final results in state
    if conclusion:  # Only store if we have a conclusion
        # Update ratings
        state = {
            **state,
            "final_ratings": {
                **state["final_ratings"],
                **ratings
            }
        }
        
        # Add new risks
        state = {
            **state,
            "final_risks": state["final_risks"] + risks
        }
        
        # Add to approved innovations if applicable
        if 'APPROVED' in conclusion.upper():
            state = {
                **state,
                "final_innovations": state["final_innovations"] + [current_round.innovation_point]
            }
    
    # Update state stage
    if state["stage"] != AgentStage.COMPLETE:
        state = update_stage(state, AgentStage.ANALYSIS)  # Back to innovation for next round
    
    return state
    
    return state
