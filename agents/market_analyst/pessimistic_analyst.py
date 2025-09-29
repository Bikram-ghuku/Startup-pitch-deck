"""
Pessimistic market analysis that focuses on risks and challenges.
"""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState

async def analyze_pessimistically(state: MarketResearchState) -> Dict:
    """
    Analyze a product from a pessimistic market perspective.
    Takes a MarketResearchState and returns state dict.
    """
    llm = ChatGroq(
        groq_api_key=state.groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=300
    )
    
    # First, determine what to research
    research_messages = [
        SystemMessage(content="""You are a pessimistic market analyst. Your goal is to identify potential 
        risks, challenges, and market barriers. While staying objective, you focus on critical analysis 
        and potential failure points that need to be addressed."""),
        HumanMessage(content=f"""What should we research to understand the risks and challenges for this product?
        
        Product Description: {state.product_description}
        
        Think step by step about what information would help identify potential problems and barriers to success.""")
    ]
    
    research_plan = await llm.ainvoke(research_messages)
    
    # Use web search based on the LLM's research plan
    from tools.web_search import web_search
    market_data = web_search(research_plan.content)
    
    # Analyze findings with a pessimistic perspective
    analysis_messages = [
        SystemMessage(content="""You are a pessimistic market analyst. Based on the research data, identify 
        critical challenges and risks. Focus on:
        
        1. Market risks and barriers to entry
        2. Competitive threats and market saturation
        3. Regulatory challenges and compliance issues
        4. Resource constraints and operational challenges
        5. Potential failure points
        
        While maintaining objectivity, ensure all significant risks are surfaced and analyzed."""),
        HumanMessage(content=f"""Analyze this market research data for our product:
        
        Product: {state.product_description}
        
        Research Findings:
        {market_data}
        
        Provide a detailed analysis that another agent could use to understand the risks and challenges.""")
    ]
    
    analysis = await llm.ainvoke(analysis_messages)
    
    # Update state
    state.pessimistic_analysis = analysis.content
    return state.dict()