"""
Pessimistic market analysis that focuses on risks and challenges.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

async def analyze_pessimistically(product_description: str, groq_api_key: str) -> str:
    """
    Analyze a product from a pessimistic market perspective.
    Returns analysis as a detailed text that can be used by other agents.
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.7
    )
    
    # First, determine what to research
    research_messages = [
        SystemMessage(content="""You are a pessimistic market analyst. Your goal is to identify potential 
        risks, challenges, and market barriers. While staying objective, you focus on critical analysis 
        and potential failure points that need to be addressed."""),
        HumanMessage(content=f"""What should we research to understand the risks and challenges for this product?
        
        Product Description: {product_description}
        
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
        
        Product: {product_description}
        
        Research Findings:
        {market_data}
        
        Provide a detailed analysis that another agent could use to understand the risks and challenges.""")
    ]
    
    analysis = await llm.ainvoke(analysis_messages)
    return analysis.content