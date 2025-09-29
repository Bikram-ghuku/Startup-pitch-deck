"""
Debater agent that critiques and challenges innovation proposals.
"""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState

async def debate_proposal(state: MarketResearchState) -> Dict:
    """
    Critique and challenge an innovation proposal.
    Takes a MarketResearchState and returns state dict.
    """
    llm = ChatGroq(
        groq_api_key=state.groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=300
    )
    
    # First, determine what to research for the critique
    research_messages = [
        SystemMessage(content="""You are a critical debater. Your goal is to identify potential flaws, 
        challenges, and improvements in innovation proposals. Focus on constructive criticism that can 
        lead to better solutions."""),
        HumanMessage(content=f"""What should we research to effectively critique this proposal?
        
        Product Description:
        {state.product_description}
        
        Innovation Proposal:
        {state.current_proposal}
        
        Think step by step about what information we need to critically evaluate this proposal.""")
    ]
    
    research_plan = await llm.ainvoke(research_messages)
    
    # Use web search based on the LLM's research plan
    from tools.web_search import web_search
    critique_data = web_search(research_plan.content)
    
    # Generate critique
    critique_messages = [
        SystemMessage(content="""You are a critical debater. Based on research and analysis, provide 
        constructive criticism of the innovation proposal. Focus on:
        
        1. Technical feasibility challenges
        2. Resource and implementation constraints
        3. Market assumption validation
        4. Potential failure modes
        5. Areas for improvement
        
        Ensure criticism is specific, actionable, and aimed at improving the proposal."""),
        HumanMessage(content=f"""Critique this innovation proposal based on the research:
        
        Product Description:
        {state.product_description}
        
        Innovation Proposal:
        {state.current_proposal}
        
        Research Findings:
        {critique_data}
        
        Limit the critique to 200 words.
        Provide a detailed critique that can be used to improve the proposal.""")
    ]
    
    state.iteration_count += 1
    critique = await llm.ainvoke(critique_messages)
    
    # Update state
    state.current_critique = critique.content
    print("\n\n\nInnovation Critique: ", critique.content)
    return state.dict()