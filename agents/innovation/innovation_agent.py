"""
Innovation agent that suggests product features and implementation approaches.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

async def generate_innovation_proposal(product_description: str, market_synthesis: str, groq_api_key: str) -> str:
    """
    Generate innovative product features and implementation approaches based on product idea and market analysis.
    Returns proposal as detailed text that can be used by other agents.
    
    Args:
        product_description: Original product idea/description
        market_synthesis: Synthesized market analysis from market research stage
        groq_api_key: API key for ChatGroq
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.7
    )
    
    # First, determine what to research
    research_messages = [
        SystemMessage(content="""You are an innovation strategist. Your goal is to propose innovative 
        product features and implementation approaches based on the original product idea and market analysis. 
        Focus on practical innovation that addresses market needs while being technically feasible."""),
        HumanMessage(content=f"""What should we research to develop innovative features for this product?
        
        Original Product Idea:
        {product_description}
        
        Market Analysis:
        {market_synthesis}
        
        Think step by step about what technical and market information we need to propose innovative solutions 
        that enhance the original product idea while addressing market needs.""")
    ]
    
    research_plan = await llm.ainvoke(research_messages)
    
    # Use web search based on the LLM's research plan
    from tools.web_search import web_search
    innovation_data = web_search(research_plan.content)
    
    # Generate innovation proposal
    proposal_messages = [
        SystemMessage(content="""You are an innovation strategist. Based on the original product idea, 
        market analysis, and research, propose innovative product features and implementation approaches. 
        Focus on:
        
        1. Core product features that enhance the original idea
        2. Technical implementation approach
        3. Innovation differentiators
        4. Development roadmap
        5. Resource requirements
        
        Ensure proposals are both innovative and practically achievable while staying true to the 
        original product vision."""),
        HumanMessage(content=f"""Generate an innovation proposal based on this information:
        
        Original Product Idea:
        {product_description}
        
        Market Analysis:
        {market_synthesis}
        
        Technical Research:
        {innovation_data}
        
        Provide a detailed proposal that another agent could critique and refine. Make sure to explain 
        how each proposed innovation enhances the original product idea while addressing market needs.""")
    ]
    
    proposal = await llm.ainvoke(proposal_messages)
    return proposal.content