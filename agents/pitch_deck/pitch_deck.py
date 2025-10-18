"""
Pitch deck generation agent that creates compelling HTML presentations.
"""
from typing import Dict
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from states.agent_state import MarketResearchState
from tools.save_pitch_deck_html import save_pitch_deck_html
from tools.image_search import image_search

async def generate_pitch_deck(state: MarketResearchState) -> Dict:
    """
    Generate an HTML pitch deck based on market analysis and innovation proposal.
    Takes a MarketResearchState and returns state dict.
    """
    llm = ChatGroq(
        groq_api_key=state.groq_api_key,
        model_name="qwen/qwen3-32b",
        temperature=0.7,
        max_tokens=8000
    )
    
    # Bind image_search tool to the LLM for autonomous tool calling
    llm_with_tools = llm.bind_tools([image_search])
    
    # Generate pitch deck HTML with autonomous image search
    messages = [
        SystemMessage(content="""Create 8-10 content-rich pitch deck slides. FOCUS ON SUBSTANTIAL TEXT CONTENT - these are full-screen slides, not image galleries.

HTML FORMAT:
```html
<div class="slide" style="background: #yourcolor; padding: 80px;">
    <h2 style="font-size: 3em; color: #textcolor; font-weight: 700; margin-bottom: 30px;">Slide Title</h2>
    <!-- Your rich content here: paragraphs, lists, tables, etc. -->
    <div class="slide-number">1</div>
</div>
```

FIXED STYLING RULES (Non-negotiable):
1. Light backgrounds (#fff, #f5f5f5, pastels) → Dark text (#000, #333, #2c3e50)
2. Dark backgrounds (#000, #1a1a2e, dark gradients) → Light text (#fff, #f0f0f0)
3. Images: Always `<img src="url" style="width: 400px; height: 300px; object-fit: cover;">`
4. NO background images. Use solid colors or gradients only.
5. All styling inline on every element.

CONTENT REQUIREMENTS (THIS IS PRIORITY #1):
Each slide needs 150-250+ words of actual content:
- Write detailed paragraphs (4-6 sentences each)
- Create bullet lists with substantial descriptions
- Add tables for comparisons/pricing with details
- Include specific data, metrics, numbers
- Explain concepts thoroughly - don't just list keywords

REQUIRED SLIDES:
1. Title - company name, tagline, key value prop (100+ words)
2. Problem - 4-5 detailed problem points with explanations
3. Market - market size, segments, growth drivers (detailed analysis)
4. Solution - how your product works (detailed explanation)
5. Features - 5-6 features with full descriptions
6. Technology - technical details and advantages
7. Business Model - revenue, pricing, economics (specific numbers)
8. Competition - competitive analysis (table or detailed points)
9. Go-to-Market - channels, strategy, partnerships (actionable details)
10. Investment - ask amount, use of funds breakdown, milestones

IMAGE USAGE:
- Use image_search 2-4 times for inline visuals
- Images are OPTIONAL - only if they enhance content
- Most slides should be text-heavy with NO images

CREATIVE FREEDOM:
- Choose any color scheme that looks professional
- Use any layout: columns, grids, cards, tables
- Add visual elements: borders, shadows, rounded corners

Generate slides with RICH, DETAILED TEXT CONTENT."""),
        HumanMessage(content=f"""Create 8-10 slides with detailed, comprehensive content based on:

PRODUCT: {state.product_description}

MARKET ANALYSIS: {state.market_synthesis}

INNOVATION: {state.current_proposal}

Write 150-250+ words per slide. Focus on explaining concepts thoroughly with specific data and examples. Use image_search for 2-4 visuals only. Prioritize text content over images.""")
    ]

    print("Content Received: ", "\nDescription: ", state.product_description, "\nMarket Analysis: ", state.market_synthesis, "\nInnovation: ", state.current_proposal)
    
    print("\n\n📝 Generating content-rich pitch deck...")
    print("🔍 Searching for supporting visuals (2-4 images max)...")
    response = await llm_with_tools.ainvoke(messages)
    
    # Check if LLM made tool calls and execute them
    tool_call_count = 0
    while hasattr(response, 'tool_calls') and response.tool_calls:
        messages.append(response)
        
        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_call_count += 1
            query = tool_call['args'].get('query', 'business')
            print(f"  📸 [{tool_call_count}] Searching: '{query}'")
            
            # Limit to 4 image searches max
            if tool_call_count <= 4:
                tool_result = image_search.invoke(query)
            else:
                print("    ⚠️  Limit reached - max 4 images")
                tool_result = '{"query": "' + query + '", "count": 0, "images": []}'
            
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call['id']
            })
        
        # Get next response from LLM
        print("✍️  Writing detailed slide content...")
        response = await llm_with_tools.ainvoke(messages)
    
    print(f"✅ Generated deck with {min(tool_call_count, 4)} images")
    
    slides_html = response.content.strip()
    
    # Count slides
    import re
    import os
    import glob
    
    slide_pattern = r'<div\s+class="slide"'
    slide_count = len(re.findall(slide_pattern, slides_html))
    
    # Count words
    text_content = re.sub(r'<style[^>]*>.*?</style>', '', slides_html, flags=re.DOTALL)
    text_content = re.sub(r'<[^>]+>', ' ', text_content)
    words = [w for w in text_content.split() if len(w) > 2]
    word_count = len(words)
    avg_words = word_count // slide_count if slide_count > 0 else 0
    
    # Count images
    img_count = len(re.findall(r'<img\s+', slides_html))
    
    print("\n📊 CONTENT ANALYSIS:")
    print(f"   Slides: {slide_count}")
    print(f"   Total words: {word_count}")
    print(f"   Avg words/slide: {avg_words}")
    print(f"   Images: {img_count}")
    
    if avg_words < 120:
        print("   ⚠️  WARNING: Content too sparse - target 150-250 words/slide")
    elif avg_words >= 150:
        print("   ✅ Excellent content depth!")
    else:
        print("   ✓  Good content, could add more detail")
    
    # Extract title
    title_match = re.search(r'<h1[^>]*>(.*?)</h1>', slides_html, re.IGNORECASE | re.DOTALL)
    presentation_title = title_match.group(1) if title_match else "Investment Pitch Deck"
    presentation_title = re.sub(r'<[^>]+>', '', presentation_title).strip()
    
    # Sequential numbering
    output_dir = "output"
    existing_files = glob.glob(os.path.join(output_dir, "pitch_deck_*.html"))
    if existing_files:
        numbers = []
        for f in existing_files:
            match = re.search(r'pitch_deck_(\d+)\.html', os.path.basename(f))
            if match:
                numbers.append(int(match.group(1)))
        next_num = max(numbers) + 1 if numbers else 1
    else:
        next_num = 1
    
    filename = f"pitch_deck_{next_num}.html"
    
    # Save
    print(f"\n💾 Saving as {filename}...")
    html_result = save_pitch_deck_html.invoke({
        "slides_html": slides_html,
        "filename": filename,
        "title": presentation_title
    })
    print(f"✅ {html_result}")
    
    # Store in state
    state.pitch_deck = slides_html
    state.pitch_deck_html = html_result
    
    return state.dict()
