"""
Pitch deck generation agent that creates compelling presentations.
Provides content in JSON format required by the PowerPoint generation tool.
"""
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
import json

async def generate_pitch_deck(product_description: str, market_analysis: str, innovation_proposal: str, groq_api_key: str) -> str:
    """
    Generate a pitch deck based on product, market analysis and innovation proposal.
    Returns pitch deck content in JSON format required by the PowerPoint tool.
    """
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.7
    )
    
    # Generate pitch deck content
    pitch_messages = [
        SystemMessage(content="""You are a pitch deck specialist. Create a compelling pitch deck that 
        tells a convincing story to investors. The output must be a valid JSON string with the following structure:
        {
            "slides": [
                {
                    "title": "slide title",
                    "content": ["bullet point 1", "bullet point 2"] or "paragraph text",
                    "notes": "optional presenter notes"
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
        
        For bullet point slides, provide content as a list of strings.
        For narrative slides, provide content as a single string.
        Include presenter notes with key talking points."""),
        HumanMessage(content=f"""Create a pitch deck based on this information:
        
        Product Description:
        {product_description}
        
        Market Analysis:
        {market_analysis}
        
        Innovation Proposal:
        {innovation_proposal}
        
        Return ONLY the JSON string with no additional text.""")
    ]
    
    pitch_content = await llm.ainvoke(pitch_messages)
    
    # Validate JSON structure
    try:
        deck = json.loads(pitch_content.content)
        if "slides" not in deck or not isinstance(deck["slides"], list):
            raise ValueError("Invalid deck structure")
        return pitch_content.content
    except Exception as e:
        # If LLM output isn't valid JSON, create a basic valid structure
        return json.dumps({
            "slides": [
                {
                    "title": "Error in Deck Generation",
                    "content": ["Failed to generate valid pitch deck structure", str(e)],
                    "notes": "Please regenerate the pitch deck"
                }
            ]
        })