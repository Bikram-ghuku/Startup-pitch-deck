"""
Debater agent that critiques and challenges innovation proposals.
"""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState
from tools.web_search import web_search

async def debate_proposal(state: MarketResearchState) -> Dict:
    """
    Critique and challenge an innovation proposal.
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
        SystemMessage(content="""You are a critical debater. Your goal is to identify potential flaws, 
        challenges, and improvements in innovation proposals. Focus on constructive criticism that can 
        lead to better solutions.
        
        You have access to a web_search tool. Use it to research validation data when needed."""),
        HumanMessage(content=f"""Critique this innovation proposal:
        
        Product Description:
        {state.product_description}
        
        Innovation Proposal:
        {state.current_proposal}
        
        Research technical feasibility and market assumptions as needed. Then provide constructive criticism focusing on:
        1. Technical feasibility challenges
        2. Resource and implementation constraints
        3. Market assumption validation
        4. Potential failure modes
        5. Areas for improvement
        
        Ensure criticism is specific, actionable, and aimed at improving the proposal. Limit the critique to 100 words.""")
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
    
    state.iteration_count += 1
    critique = response

    print("\n\n\nInnovation Critique: ", critique.content)
    
    # Update state
    state.current_critique = critique.content
    return state.dict()