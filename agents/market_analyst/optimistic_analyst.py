"""
Optimistic market analysis that focuses on opportunities and growth potential.
"""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState
from tools.web_search import web_search

async def analyze_optimistically(state: MarketResearchState) -> Dict:
    """
    Analyze a product from an optimistic market perspective.
    Takes a MarketResearchState and returns state dict.
    """
    llm = ChatGroq(
        groq_api_key=state.groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1500
    )
    
    # Bind web_search tool to the LLM for autonomous tool calling
    llm_with_tools = llm.bind_tools([web_search])
    
    # Let the LLM autonomously decide when to use web search
    messages = [
        SystemMessage(content="""You are an optimistic market analyst. Your goal is to identify promising 
        market opportunities and growth potential. While staying grounded in reality, you focus on positive 
        indicators and potential for success.
        
        You have access to a web_search tool. Use it to research information when needed."""),
        HumanMessage(content=f"""Analyze the market potential for this product:
        
        Product Description: {state.product_description}
        
        Research the market opportunities, customer needs, and growth potential. Then provide a detailed analysis focusing on:
        1. Market opportunities and growth indicators
        2. Target customer segments and their needs
        3. Competitive advantages
        4. Market timing and trends
        5. Potential for success
        
        While maintaining credibility, emphasize positive signals and growth potential.""")
    ]
    
    response = await llm_with_tools.ainvoke(messages)
    
    # Check if LLM made tool calls and execute them
    while hasattr(response, 'tool_calls') and response.tool_calls:
        messages.append(response)
        
        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_result = web_search.invoke(tool_call['args']['query'])
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call['id']
            })
        
        # Get next response from LLM
        response = await llm_with_tools.ainvoke(messages)
    
    analysis = response
    
    # Update state
    state.optimistic_analysis = analysis.content
    return state.dict()