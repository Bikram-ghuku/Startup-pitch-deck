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
        model_name="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=8000
    )
    
    # Bind image_search tool to the LLM for autonomous tool calling
    llm_with_tools = llm.bind_tools([image_search])
    
    # Generate pitch deck HTML with autonomous image search
    messages = [
        SystemMessage(content="""You are an expert pitch deck designer. Your PRIMARY focus is CONTENT, not images.

CRITICAL IMAGE RULES (STRICTLY ENFORCE):
- Maximum 0-1 images PER SLIDE (never 2 or more)
- Only 3-4 slides total should have images (out of 8-10 slides)
- If you use an image, it should be SMALL - max 25% of slide space
- Most slides should be TEXT ONLY with NO images
- Images are OPTIONAL decoration, content is MANDATORY

MANDATORY TOOL USAGE:
Use image_search tool 2-3 times MAXIMUM for the entire deck (not per slide):
- One hero/title image (e.g., "ai technology", "startup office")
- One problem/solution visual (e.g., "business challenge", "innovation")
- One team/growth visual (e.g., "team collaboration", "growth chart")

Use SHORT 2-3 word queries only.

FIXED COLOR SCHEME (use exactly these):
- Gradient background: linear-gradient(135deg, #667eea 0%, #764ba2 100%)
- Accent gradient: linear-gradient(135deg, #f093fb 0%, #f5576c 100%)
- Card background: #f8f9fa
- Text colors: #2c3e50 (headings), #4a5568 (body)
- Border accent: #667eea

TYPOGRAPHY RULES:
- H2: font-size: 3em; color: #2c3e50; font-weight: 800; margin-bottom: 40px;
- H3: font-size: 2em; color: #2c3e50; font-weight: 700; margin-bottom: 20px;
- Body: font-size: 1.4em; line-height: 1.8; color: #4a5568;
- All text: font-family: 'Arial', sans-serif;

MANDATORY CONTENT STRUCTURE:

Each slide MUST follow this pattern with MINIMUM word counts:

**Slide 1: Title (50 words minimum)**
- Company name (large)
- Compelling tagline (15-20 words)
- Key differentiator or metric
- NO IMAGE or small background image only

**Slide 2: Problem (200+ words)**
Structure with 4-5 boxes/cards:
```html
<div style="background: #f8f9fa; padding: 30px; border-radius: 12px; border-left: 5px solid #667eea; margin-bottom: 25px;">
    <h3 style="font-size: 2em; color: #2c3e50; margin-bottom: 15px; font-weight: 700;">Problem Title</h3>
    <p style="font-size: 1.4em; line-height: 1.8; color: #4a5568;">Detailed 3-4 sentence explanation of the problem, its impact, who it affects, and why current solutions fail. Include specific pain points and consequences.</p>
</div>
```
Repeat 4-5 times. NO IMAGES.

**Slide 3: Market Opportunity (250+ words)**
Three sections required:
1. Market Size box with $ amount and 2-3 sentences explanation
2. Target Segments box with detailed demographics (3-4 sentences)
3. Growth Drivers box with 3-4 driving factors (3-4 sentences each)
Use colored gradient boxes. NO IMAGES.

**Slide 4: Solution (200+ words)**
Layout: 
- Top: Overview paragraph (4-5 sentences explaining core solution)
- Below: 3-4 solution boxes addressing each problem
Each box needs: Title + 3-4 sentence description
ONE SMALL IMAGE optional (max 200px height, positioned at top right)

**Slide 5: Product Features (250+ words)**
5-6 feature cards in grid:
```html
<div style="background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); border-top: 4px solid #667eea;">
    <h3 style="font-size: 1.8em; color: #2c3e50; margin-bottom: 15px;">Feature Name</h3>
    <p style="font-size: 1.3em; line-height: 1.8; color: #4a5568;">3-4 sentences describing the feature, how it works, and specific benefits to users. Include technical details.</p>
</div>
```
NO IMAGES.

**Slide 6: Technology/Innovation (200+ words)**
3 sections:
1. Core Technology (3-4 sentences)
2. Proprietary Advantages (3-4 sentences)
3. Defensibility/Moat (3-4 sentences)
Use card layouts. NO IMAGES.

**Slide 7: Business Model (250+ words)**
Must include:
1. Revenue Streams (2-3 streams with 2-3 sentences each)
2. Pricing Structure (specific tiers with prices and descriptions)
3. Unit Economics (CAC, LTV, margins with explanations)
4. Customer Acquisition (2-3 channels with detailed strategies)
NO IMAGES.

**Slide 8: Competition (200+ words)**
Either comparison table OR 4-5 differentiation cards:
```html
<div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 30px; border-radius: 12px; color: white; margin-bottom: 20px;">
    <h3 style="font-size: 1.8em; margin-bottom: 15px;">Advantage Title</h3>
    <p style="font-size: 1.3em; line-height: 1.8;">3-4 sentences explaining the specific advantage, why competitors can't replicate it, and the impact on customer choice.</p>
</div>
```
NO IMAGES.

**Slide 9: Go-to-Market (200+ words)**
4 sections required:
1. Customer Acquisition Channels (3-4 channels with strategies)
2. Partnerships (2-3 partner types with specific names/approaches)
3. Marketing Strategy (specific tactics and timeline)
4. Sales Process (step-by-step with conversion expectations)
ONE SMALL IMAGE optional

**Slide 10: Investment Ask (200+ words)**
Must include:
1. Amount seeking (prominent)
2. Use of funds breakdown (5-6 categories with amounts and detailed explanations)
3. Milestones (3-4 specific milestones with timelines)
4. Expected outcomes (metrics and impact)
NO IMAGES.

LAYOUT RULES:
- padding: 70px 90px; on all slides
- Vertical layouts ONLY (no side-by-side unless image is tiny corner element)
- If image used: position in top-right corner with max 200px width
- Content fills 100% width
- Cards/boxes should stack vertically
- Generous spacing: margin-bottom: 25px; between elements

EXAMPLE PERFECT SLIDE (Problem - text only, 250+ words):

```html
<div class="slide" style="background: white; padding: 70px 90px;" data-notes="Present detailed problems">
    <h2 style="font-size: 3em; color: #2c3e50; font-weight: 800; margin-bottom: 50px; border-bottom: 5px solid #667eea; padding-bottom: 20px;">The Problem We're Solving</h2>
    
    <div style="background: #f8f9fa; padding: 30px; border-radius: 12px; border-left: 5px solid #667eea; margin-bottom: 25px;">
        <h3 style="font-size: 2em; color: #2c3e50; margin-bottom: 15px; font-weight: 700;">Fragmented Financial Data</h3>
        <p style="font-size: 1.4em; line-height: 1.8; color: #4a5568;">Modern consumers have their finances scattered across 5-7 different platforms: banking apps, credit cards, investment accounts, and payment services. This fragmentation makes it nearly impossible to understand their complete financial picture. Users waste 3-4 hours per week manually tracking expenses across platforms, leading to poor financial decisions and missed savings opportunities worth an average of $2,400 annually per user.</p>
    </div>
    
    <div style="background: #f8f9fa; padding: 30px; border-radius: 12px; border-left: 5px solid #f5576c; margin-bottom: 25px;">
        <h3 style="font-size: 2em; color: #2c3e50; margin-bottom: 15px; font-weight: 700;">Complex Existing Solutions</h3>
        <p style="font-size: 1.4em; line-height: 1.8; color: #4a5568;">Current financial management tools overwhelm users with excessive features and complicated interfaces designed for financial professionals. Studies show 73% of users abandon financial apps within the first month because they're too complex to understand and require significant time investment to set up properly. The learning curve prevents mass adoption and limits the market to finance-savvy individuals.</p>
    </div>
    
    <div style="background: #f8f9fa; padding: 30px; border-radius: 12px; border-left: 5px solid #667eea; margin-bottom: 25px;">
        <h3 style="font-size: 2em; color: #2c3e50; margin-bottom: 15px; font-weight: 700;">Lack of Personalization</h3>
        <p style="font-size: 1.4em; line-height: 1.8; color: #4a5568;">Generic financial advice doesn't account for individual circumstances, goals, or life stages. A 25-year-old starting their career has vastly different needs than a 45-year-old planning retirement, yet existing apps provide one-size-fits-all recommendations. This results in irrelevant insights that users ignore, missing opportunities to genuinely improve their financial health through tailored, actionable advice.</p>
    </div>
    
    <div style="background: #f8f9fa; padding: 30px; border-radius: 12px; border-left: 5px solid #f5576c; margin-bottom: 25px;">
        <h3 style="font-size: 2em; color: #2c3e50; margin-bottom: 15px; font-weight: 700;">No Proactive Guidance</h3>
        <p style="font-size: 1.4em; line-height: 1.8; color: #4a5568;">Current tools are reactive, showing what happened in the past without helping users make better future decisions. Users need proactive alerts about upcoming bills, spending pattern warnings, and automated optimization suggestions. Without predictive intelligence, users repeatedly make the same financial mistakes, accumulating debt and missing investment opportunities that could compound into significant wealth over time.</p>
    </div>
    
    <div class="slide-number">2</div>
</div>
```

CRITICAL SUCCESS CRITERIA:
✓ Each slide has 200-250+ words of actual content
✓ Maximum 3-4 images in the ENTIRE deck (not per slide)
✓ All text properly styled and aligned
✓ Content is specific, detailed, data-driven
✓ No generic statements - everything backed by numbers or specifics
✓ Consistent color scheme throughout

Generate ONLY <div class="slide"> elements."""),
        HumanMessage(content=f"""Create a content-rich pitch deck (8-10 slides) based on:

PRODUCT:
{state.product_description}

MARKET ANALYSIS:
{state.market_synthesis}

INNOVATION:
{state.current_proposal}

REQUIREMENTS:
1. Use image_search ONLY 2-3 times total for: product category (2 words), one problem/solution visual (2 words), one growth visual (2 words)
2. Generate 8-10 slides following the exact structure above
3. MOST slides should have ZERO images - focus on rich text content
4. If a slide has an image, make it small (200px max) in top-right corner
5. Each slide MUST have 200-250+ words minimum
6. Use the exact color scheme and typography provided
7. Follow the card/box structure for organizing content
8. Include specific numbers, metrics, data points throughout
9. Every statement should be detailed with 3-4 sentence explanations

Focus on CONTENT DENSITY over visual design. Text is priority #1.""")
    ]
    
    print("\n\n🎨 Starting content-focused pitch deck generation...")
    print("📸 Phase 1: Limited image search (2-3 images max)...")
    response = await llm_with_tools.ainvoke(messages)
    
    # Check if LLM made tool calls and execute them
    tool_call_count = 0
    while hasattr(response, 'tool_calls') and response.tool_calls:
        messages.append(response)
        
        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_call_count += 1
            query = tool_call['args'].get('query', 'business')
            print(f"  🔍 [{tool_call_count}] Searching: '{query}'")
            
            # Limit to 3 image searches max
            if tool_call_count <= 3:
                tool_result = image_search.invoke(query)
            else:
                print("    ⚠️  Skipping - max 3 images enforced")
                tool_result = '{"query": "' + query + '", "count": 0, "images": []}'
            
            messages.append({
                "role": "tool",
                "content": tool_result,
                "tool_call_id": tool_call['id']
            })
        
        # Get next response from LLM
        print("📝 Phase 2: Generating text-heavy slides...")
        response = await llm_with_tools.ainvoke(messages)
    
    print(f"✅ Used {min(tool_call_count, 3)} images (max 3 enforced)")
    
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
    
    print("\n📊 DECK STATISTICS:")
    print(f"   Slides: {slide_count}")
    print(f"   Total words: {word_count}")
    print(f"   Avg words/slide: {avg_words}")
    print(f"   Images used: {img_count}")
    
    if avg_words < 180:
        print("   ⚠️  WARNING: Content density too low! Target is 200-250 words/slide")
    else:
        print("   ✅ Content density is good!")
    
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
