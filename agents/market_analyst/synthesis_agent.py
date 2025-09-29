"""
Synthesis agent that combines and balances optimistic and pessimistic analyses.
"""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState

async def synthesize_analysis(state: MarketResearchState) -> Dict:
    """
    Synthesize optimistic and pessimistic market analyses into a balanced view.
    Takes a MarketResearchState and returns state dict.
    """
    llm = ChatGroq(
        groq_api_key=state.groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.7
    )
    
    synthesis_messages = [
        SystemMessage(content="""You are a synthesis analyst responsible for combining optimistic and 
        pessimistic market analyses into a balanced, comprehensive view. Your goal is to:
        
        1. Identify areas of agreement between perspectives
        2. Highlight key tensions and trade-offs
        3. Provide balanced recommendations
        4. Ensure all significant points from both analyses are considered
        
        Create a nuanced analysis that acknowledges both opportunities and challenges."""),
        HumanMessage(content=f"""Synthesize these two market analyses into a balanced perspective:
        
        Product Description:
        {state.product_description}
        
        Optimistic Analysis:
        {state.optimistic_analysis}
        
        Pessimistic Analysis:
        {state.pessimistic_analysis}
        
        Provide a comprehensive synthesis that another agent could use to understand the complete market picture.""")
    ]
    
    synthesis = await llm.ainvoke(synthesis_messages)
    
    # Update state
    state.market_synthesis = synthesis.content
    return state.dict()