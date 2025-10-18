"""
Pessimistic market analysis that focuses on risks and challenges.
"""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState
from tools.web_search import web_search

async def analyze_pessimistically(state: MarketResearchState) -> Dict:
    """
    Analyze a product from a pessimistic market perspective.
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
        SystemMessage(content="""You are a pessimistic market analyst. Your goal is to identify potential 
        risks, challenges, and market barriers. While staying objective, you focus on critical analysis 
        and potential failure points that need to be addressed.
        
        You have access to a web_search tool. Use it to research information when needed."""),
        HumanMessage(content=f"""Analyze the risks and challenges for this product:
        
        Product Description: {state.product_description}
        
        Research the market, competitors, and potential barriers. Then provide a detailed analysis focusing on:
        1. Market risks and barriers to entry
        2. Competitive threats and market saturation
        3. Regulatory challenges and compliance issues
        4. Resource constraints and operational challenges
        5. Potential failure points
        
        Ensure all significant risks are surfaced and analyzed.""")
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
    state.pessimistic_analysis = analysis.content
    return state.dict()