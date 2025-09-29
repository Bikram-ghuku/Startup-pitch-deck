"""
Synthesis agent that combines and balances optimistic and pessimistic analyses.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

async def synthesize_analysis(optimistic_analysis: str, pessimistic_analysis: str, groq_api_key: str) -> str:
    """
    Synthesize optimistic and pessimistic market analyses into a balanced view.
    Returns analysis as a detailed text that can be used by other agents.
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
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
        
        Optimistic Analysis:
        {optimistic_analysis}
        
        Pessimistic Analysis:
        {pessimistic_analysis}
        
        Provide a comprehensive synthesis that another agent could use to understand the complete market picture.""")
    ]
    
    synthesis = await llm.ainvoke(synthesis_messages)
    return synthesis.content
