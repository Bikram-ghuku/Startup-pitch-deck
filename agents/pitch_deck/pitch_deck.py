"""
Pitch deck generation agent that creates compelling presentations.
"""
from typing import Dict
import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState
from tools.save_pitch_deck_ppt import save_pitch_deck_ppt

async def generate_pitch_deck(state: MarketResearchState) -> Dict:
    """
    Generate a pitch deck based on market analysis and innovation proposal.
    Takes a MarketResearchState and returns state dict.
    """
    llm = ChatGroq(
        groq_api_key=state.groq_api_key,
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=1024
    )
    
    # Generate pitch deck content
    pitch_messages = [
        SystemMessage(content="""You are a pitch deck specialist. Create a compelling pitch deck that 
        tells a convincing story to investors. The output must be a valid JSON string with the following structure:
        Do not assume anything, just generate the pitch deck based on the information provided.
        {
            "slides": [
                {
                    "title": "slide title",
                    "content": ["bullet point 1", "bullet point 2"] or "paragraph text",
                    "notes": "optional presenter notes",
                    "image": "Prompts to generate an image for the slide"
                }
            ]
        }
        
        Structure the deck to include:
        1. Title & Vision
        2. Problem Statement
        3. Market Opportunity
        4. Solution & Innovation
        5. Competitive Analysis
        6. Business Model
        7. Go-to-Market Strategy
        8. Team & Execution
        9. Investment Ask
        
        For bullet point slides, provide content as a list of strings having atleast 3 bullet points and atmost 7 bullet points.
        For narrative slides, provide content as a single string.
        Include presenter notes with key talking points.
        Include an image for each slide.
        """),
        HumanMessage(content=f"""Create a pitch deck based on this information:
        
        Product Description:
        {state.product_description}
        
        Market Analysis:
        {state.market_synthesis}
        
        Innovation Proposal:
        {state.current_proposal}
        
        Return ONLY the JSON string with no additional text.
        If the pitch deck is not generated, return the error message in the content of the slide.""")
    ]
    
    pitch_content = await llm.ainvoke(pitch_messages)
    
    # Validate JSON structure
    print("\n\n\nPitch Deck: ", pitch_content.content)
    try:
        deck = json.loads(pitch_content.content)
        if "slides" not in deck or not isinstance(deck["slides"], list):
            raise ValueError("Invalid deck structure")
        state.pitch_deck = pitch_content.content
        pitch_deck_result = save_pitch_deck_ppt(pitch_content.content)
        print("\n\n\nPitch Deck Result: ", pitch_deck_result)
    except Exception as e:
        # If LLM output isn't valid JSON, create a basic valid structure
        state.pitch_deck = json.dumps({
            "slides": [
                {
                    "title": "Error in Deck Generation",
                    "content": ["Failed to generate valid pitch deck structure", str(e)],
                    "notes": "Please regenerate the pitch deck",
                    "image": "Error in Deck Generation"
                }
            ]
        })
    
    return state.dict()