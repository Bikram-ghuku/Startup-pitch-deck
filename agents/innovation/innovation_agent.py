"""
Innovation agent that suggests product features and implementation approaches.
"""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState
from tools.web_search import web_search

async def generate_innovation_proposal(state: MarketResearchState) -> Dict:
    """
    Generate innovative product features and implementation approaches.
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
        SystemMessage(content="""You are an innovation strategist. Your goal is to propose innovative 
        product features and implementation approaches based on the original product idea and market analysis. 
        Focus on practical innovation that addresses market needs while being technically feasible.
        
        You have access to a web_search tool. Use it to research technical solutions and innovations when needed."""),
        HumanMessage(content=f"""Generate an innovation proposal for this product:
        
        Product Description:
        {state.product_description}
        
        Market Analysis:
        {state.market_synthesis}

        Last Critique:
        {state.current_critique}

        Last Innovation Proposal:
        {state.current_proposal}
        
        Research technical innovations and market trends as needed. Then propose innovative features focusing on:
        1. Core product features that enhance the original idea
        2. Highly defined features that are not already in the market
        3. Features that can be added to improve its value proposition and are feasible to implement
        4. High level description of the features and how they can be implemented
        
        Address any points from the last critique. Limit the proposal to 500 words.""")
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
    
    proposal = response

    state.current_proposal = proposal.content
    return state.dict()