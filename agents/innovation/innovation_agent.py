from langchain_groq import ChatGroq
from dotenv import load_dotenv
from tools.web_search import web_search
from states.agent_state import (
    AgentState, AgentStage,
    add_search, add_innovation_debate, update_stage
)
import time

load_dotenv()

_innovation_llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.8,
    max_tokens=1024
)

def innovation_strategist(state: AgentState) -> AgentState:
    """Generate innovative features/differentiators using direct LLM invocation with web research.

    Args:
        context: Brief description of product, target users, constraints.
        goal: What to optimize for in the ideation.

    Returns:
        Structured reasoning chain and innovative suggestions with rationale.
    """
    # Get relevant state information
    current_analysis = state["current_analysis"].get('latest_insight', '')
    approved_innovations = "\n".join(state["final_innovations"]) if state["final_innovations"] else "None yet"
    current_risks = "\n".join(state["final_risks"]) if state["final_risks"] else "None identified"
    
    # Get innovation debate history
    debate_history = "\n".join([
        f"Innovation: {round.innovation_point}\n"
        f"Critique: {round.debate_critique}\n"
        f"Conclusion: {round.conclusion}\n---"
        for round in state["innovation_debate_history"][-2:]  # Last 2 rounds
    ]) if state["innovation_debate_history"] else ""
    
    prompt = (
        f"You are an innovation strategist analyzing:\n{state['context']}\n\n"
        f"CURRENT STATE:\n"
        f"Stage: {state['stage'].value}\n"
        f"Previously Approved Innovations:\n{approved_innovations}\n"
        f"Known Risks:\n{current_risks}\n"
        f"Latest Analysis:\n{current_analysis}\n\n"
        
        f"Previous Debate History:\n{debate_context}\n\n"
        
        "TOOL USAGE:\n"
        "When you need to research, use the following format:\n"
        "SEARCH: your specific search query\n\n"
        "Example searches:\n"
        "SEARCH: innovative features in smart home automation\n"
        "SEARCH: emerging AI personalization technologies\n"
        "SEARCH: unique smart home assistant capabilities\n\n"
        
        "ANALYSIS FRAMEWORK:\n"
        "1. Context Review (use search evidence)\n"
        "- Market dynamics and trends\n"
        "- Current solution landscape\n"
        "- Unmet needs and opportunities\n"
        "- Consider previous innovations to avoid repetition\n\n"
        
        "2. Innovation Pathways\n"
        "A) Evolutionary Track\n"
        "   - Enhance existing solutions\n"
        "   - Quick wins and improvements\n"
        "   - Integration opportunities\n"
        "B) Revolutionary Track\n"
        "   - Novel approaches\n"
        "   - Emerging tech applications\n"
        "   - Paradigm shifts\n\n"
        
        "3. Critical Evaluation\n"
        "- Technical feasibility check\n"
        "- Market potential assessment\n"
        "- Resource requirements\n"
        "- Risk awareness (especially known risks)\n"
        "- Differentiation from previous innovations\n\n"
        
        "4. Innovation Proposals (2-3)\n"
        "For each innovation:\n"
        "- Clear value proposition\n"
        "- Implementation approach\n"
        "- Competitive advantage\n"
        "- Risk factors\n"
        "- Success metrics\n\n"
        
        "IMPORTANT:\n"
        "- Build upon previous successes\n"
        "- Address known risks\n"
        "- Avoid repeating rejected ideas\n"
        "- Support claims with evidence\n"
        "- Be specific and actionable\n\n"
        
        "Format each section with clear headings. Focus on novel insights that build upon our existing knowledge."
    )
    def _handle_search(response_text: str, state: AgentState) -> tuple[str, AgentState]:
        """Process LLM response and handle any search requests."""
        lines = response_text.split('\n')
        result = []
        for line in lines:
            line = line.strip()
            if line.startswith('SEARCH:'):
                query = line[7:].strip()
                # Call web_search tool through state management
                state = add_search(state, query)
                search_result = state['search_results'][-1] if state['search_results'] else 'No results'
                result.append(f"Search Results for '{query}':")
                result.append(search_result)
                result.append("")
            else:
                result.append(line)
        return '\n'.join(result), state

    # Format prompt with debate history
    prompt = prompt.replace("{debate_context}", debate_history)

    # Initial response
    response = _innovation_llm.invoke(prompt)
    response_text = getattr(response, "content", str(response))

    state = {
        **state,
        "current_response": response_text
    }
    
    # Handle any searches and get final response
    final_response, state = _handle_search(response_text, state)
    
    # Store innovation point for debate
    if state["stage"] == AgentStage.ANALYSIS:
        state = add_innovation_debate(
            state=state,
            innovation=final_response,
            critique="",  # Will be filled by debater
            conclusion=""  # Will be filled after debate
        )
    
    # Update state stage
    if state["stage"] != AgentStage.COMPLETE:
        state = update_stage(state, AgentStage.DEBATE)
    
    return state