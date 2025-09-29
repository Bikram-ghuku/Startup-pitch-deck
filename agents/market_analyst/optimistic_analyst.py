"""
Optimistic market analysis that focuses on opportunities and growth potential.
"""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState

async def analyze_optimistically(state: MarketResearchState) -> Dict:
    """
    Analyze a product from an optimistic market perspective.
    Takes a MarketResearchState and returns state dict.
    """
    llm = ChatGroq(
        groq_api_key=state.groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.7
    )
    
    # First, determine what to research
    research_messages = [
        SystemMessage(content="""You are an optimistic market analyst. Your goal is to identify promising 
        market opportunities and growth potential. While staying grounded in reality, you focus on positive 
        indicators and potential for success."""),
        HumanMessage(content=f"""What should we research to understand the market potential for this product?
        
        Product Description: {state.product_description}
        
        Think step by step about what information would help build a strong case for this product's success.""")
    ]
    
    research_plan = await llm.ainvoke(research_messages)
    
    # Use web search based on the LLM's research plan
    from tools.web_search import web_search
    market_data = web_search(research_plan.content)
    
    # Analyze findings with an optimistic perspective
    analysis_messages = [
        SystemMessage(content="""You are an optimistic market analyst. Based on the research data, build a 
        compelling case for the product's market potential. Focus on:
        
        1. Market opportunities and growth indicators
        2. Target customer segments and their needs
        3. Competitive advantages
        4. Market timing and trends
        5. Potential for success
        
        While maintaining credibility, emphasize positive signals and growth potential."""),
        HumanMessage(content=f"""Analyze this market research data for our product:
        
        Product: {state.product_description}
        
        Research Findings:
        {market_data}
        
        Provide a detailed analysis that another agent could use to understand the market opportunity.""")
    ]
    
    analysis = await llm.ainvoke(analysis_messages)
    
    # Update state
    state.optimistic_analysis = analysis.content
    return state.dict()