"""
Pitch deck generation agent that creates compelling HTML presentations.
"""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState
from tools.save_pitch_deck_html import save_pitch_deck_html

async def generate_pitch_deck(state: MarketResearchState) -> Dict:
    """
    Generate an HTML pitch deck based on market analysis and innovation proposal.
    Takes a MarketResearchState and returns state dict.
    """
    llm = ChatGroq(
        groq_api_key=state.groq_api_key,
        model_name="llama-3.1-8b-instant",  # Using larger model for better HTML generation
        temperature=0.7,
        max_tokens=8000  # More tokens for HTML generation
    )
    
    # Generate pitch deck HTML
    pitch_messages = [
        SystemMessage(content="""You are an expert pitch deck designer who creates stunning HTML presentations.

Generate individual HTML slide elements for a compelling investor pitch deck. Each slide is a full-screen <div class="slide"> that takes up the ENTIRE viewport (100% width and height).

CRITICAL DESIGN RULES:
1. Use INLINE STYLES for everything - you have full creative control
2. Each slide should have its own unique design, colors, and layout
3. Use the full screen - slides are already position:absolute and 100% width/height
4. Be creative with layouts: flexbox, grid, absolute positioning
5. Use gradients, colors, typography, spacing creatively
6. Include data-notes attribute for presenter notes
7. Add slide number: <div class="slide-number">X</div>

STYLING APPROACH:
- Apply styles directly: style="background: linear-gradient(...); padding: 80px; display: flex; ..."
- Design each slide uniquely based on its content and message
- Use bold typography, impactful colors, creative layouts
- Make visual hierarchy clear through size, color, spacing

EXAMPLE SLIDES:

<div class="slide" style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); display: flex; flex-direction: column; justify-content: center; align-items: center; text-align: center; padding: 80px;" data-notes="Welcome investors with bold vision statement">
    <h1 style="font-size: 5em; font-weight: 900; color: #fff; margin-bottom: 30px; letter-spacing: -2px;">AI Finance Revolution</h1>
    <p style="font-size: 2em; color: rgba(255,255,255,0.8); max-width: 800px;">Empowering 100M Users with Intelligent Money Management</p>
    <div class="slide-number">1</div>
</div>

<div class="slide" style="background: #fff; padding: 80px 100px; display: flex; flex-direction: column;" data-notes="Present the critical problem">
    <h2 style="font-size: 3.5em; color: #2d3748; margin-bottom: 50px; border-bottom: 4px solid #e53e3e; padding-bottom: 20px;">The Problem</h2>
    <ul style="list-style: none; padding: 0;">
        <li style="font-size: 1.8em; color: #4a5568; margin-bottom: 30px; padding-left: 40px; position: relative;">
            <span style="position: absolute; left: 0; color: #e53e3e; font-size: 1.5em;">●</span>
            <strong style="color: #2d3748;">70% of millennials</strong> struggle with personal finance
        </li>
        <li style="font-size: 1.8em; color: #4a5568; margin-bottom: 30px; padding-left: 40px; position: relative;">
            <span style="position: absolute; left: 0; color: #e53e3e; font-size: 1.5em;">●</span>
            Traditional apps are complex and <strong style="color: #2d3748;">unintuitive</strong>
        </li>
    </ul>
    <div class="slide-number">2</div>
</div>

STRUCTURE (8-10 slides):
1. Title/Vision slide (full-screen, dramatic)
2. Problem (clear, impactful)
3. Market Opportunity (data-driven)
4. Solution (show the innovation)
5. Product Features (visual, organized)
6. Business Model (clear value)
7. Competitive Advantage (differentiation)
8. Go-to-Market (strategy)
9. Investment Ask (strong close)

Generate ONLY the HTML <div class="slide"> elements. Each slide should be a complete, self-contained design with inline styles."""),
        HumanMessage(content=f"""Create HTML slides for a pitch deck based on:

PRODUCT:
{state.product_description}

MARKET ANALYSIS:
{state.market_synthesis}

INNOVATION:
{state.current_proposal}

Generate professional, visually engaging HTML slides. Make each slide beautiful and impactful. Use colors, formatting, and layout strategically.""")
    ]
    
    print("\n\nGenerating HTML pitch deck...")
    pitch_content = await llm.ainvoke(pitch_messages)
    
    slides_html = pitch_content.content.strip()
    
    # Count slides
    import re
    slide_pattern = r'<div\s+class="slide\s'
    slide_count = len(re.findall(slide_pattern, slides_html))
    print(f"\nGenerated {slide_count} slides")
    
    # Extract title from first slide for the presentation title
    title_start = slides_html.find('<div class="slide-title">') + len('<div class="slide-title">')
    title_end = slides_html.find('</div>', title_start)
    presentation_title = slides_html[title_start:title_end] if title_start > 0 and title_end > 0 else "Investment Pitch Deck"
    
    # Save HTML presentation
    print("\nSaving HTML pitch deck...")
    html_result = save_pitch_deck_html.invoke({
        "slides_html": slides_html,
        "filename": "pitch_deck.html",
        "title": presentation_title
    })
    print(f"Result: {html_result}")
    
    # Store in state
    state.pitch_deck = slides_html  # Store the HTML
    state.pitch_deck_html = html_result
    
    return state.dict()