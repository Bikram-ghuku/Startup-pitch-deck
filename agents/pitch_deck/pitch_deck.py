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
        max_tokens=4000
    )
    
    # Bind image_search tool to the LLM for autonomous tool calling
    llm_with_tools = llm.bind_tools([image_search])
    
    print("Content Received: ", "\nDescription: ", state.product_description, "\nMarket Analysis: ", state.market_synthesis, "\nInnovation: ", state.current_proposal)
    
    # Step 1: Generate a high-level plan
    print("\n\n📋 Step 1: Creating slide structure plan...")
    plan_messages = [
        SystemMessage(content="""Create a detailed plan for a pitch deck with 8-10 slides. For each slide, specify:
1. Slide number and title
2. Key content points to cover (3-5 bullet points)
3. Suggested layout approach
4. Whether an image would enhance the slide

Return ONLY a structured plan like this:
SLIDE 1: [Title]
- Content point 1
- Content point 2
- Content point 3
- Layout: [approach]
- Image needed: [yes/no]

SLIDE 2: [Title]
...

Focus on creating a logical flow from problem to solution to market to business model."""),
        HumanMessage(content=f"""Create a pitch deck plan based on:

PRODUCT: {state.product_description}

MARKET ANALYSIS: {state.market_synthesis}

INNOVATION: {state.current_proposal}

Plan 8-10 slides with detailed content points for each.""")
    ]
    
    plan_response = await llm.ainvoke(plan_messages)
    slide_plan = plan_response.content.strip()
    print("✅ Plan created")
    print(f"\n📋 SLIDE PLAN:\n{slide_plan}\n")
    
    # Step 2: Generate slides iteratively
    print("\n🎨 Step 2: Generating slides iteratively...")
    all_slides_html = []
    image_count = 0
    
    # Parse the plan to extract slide information
    import re
    slide_sections = re.split(r'SLIDE \d+:', slide_plan)
    
    for i, section in enumerate(slide_sections[1:], 1):  # Skip first empty section
        print(f"\n📝 Generating Slide {i}...")
        
        # Extract slide title and content from the section
        lines = section.strip().split('\n')
        slide_title = lines[0].strip()
        
        slide_messages = [
            SystemMessage(content="""Generate ONE slide with rich content. Follow these rules:
                        HTML FORMAT:
                        ```html
                        <div class="slide" style="background: #yourcolor; padding: 80px;">
                            <h2 style="font-size: 3em; color: #textcolor; font-weight: 700; margin-bottom: 30px;">Slide Title</h2>
                            <!-- Your rich content here: paragraphs, lists, tables, etc. -->
                            <div class="slide-number">1</div>
                        </div>
                        ```

                        STYLING RULES:
                        1. Light backgrounds (#fff, #f5f5f5, pastels) → Dark text (#000, #333, #2c3e50)
                        2. Dark backgrounds (#000, #1a1a2e, dark gradients) → Light text (#fff, #f0f0f0)
                        3. Images: `<img src="url" style="width: 400px; height: 300px; object-fit: cover;">`
                        4. NO background images. Use solid colors or gradients only.
                        5. All styling inline on every element.

                        CONTENT REQUIREMENTS:
                        - Write 150-250+ words of actual content
                        - Use detailed paragraphs (4-6 sentences each)
                        - Include specific data, metrics, numbers where possible
                        - Explain concepts thoroughly

                        Generate ONLY the HTML for ONE slide."""),
                                    HumanMessage(content=f"""Generate slide {i} with title: "{slide_title}"

                        Based on this information:
                        PRODUCT: {state.product_description}
                        MARKET ANALYSIS: {state.market_synthesis}
                        INNOVATION: {state.current_proposal}

                        Slide content points from plan:
                        {section}

                        Write detailed, comprehensive content for this single slide.""")
        ]
        
        # Check if this slide needs an image
        if "Image needed: yes" in section.lower():
            slide_messages[0] = SystemMessage(content="""Generate ONE slide with rich content. Follow these rules:

                HTML FORMAT:
                ```html
                <div class="slide" style="background: #yourcolor; padding: 80px;">
                    <h2 style="font-size: 3em; color: #textcolor; font-weight: 700; margin-bottom: 30px;">Slide Title</h2>
                    <!-- Your rich content here: paragraphs, lists, tables, etc. -->
                    <div class="slide-number">1</div>
                </div>
                ```

                STYLING RULES:
                1. Light backgrounds (#fff, #f5f5f5, pastels) → Dark text (#000, #333, #2c3e50)
                2. Dark backgrounds (#000, #1a1a2e, dark gradients) → Light text (#fff, #f0f0f0)
                3. Images: `<img src="url" style="width: 400px; height: 300px; object-fit: cover;">`
                4. NO background images. Use solid colors or gradients only.
                5. All styling inline on every element.

                CONTENT REQUIREMENTS:
                - Write 150-250+ words of actual content
                - Use detailed paragraphs (4-6 sentences each)
                - Include specific data, metrics, numbers where possible
                - Explain concepts thoroughly
                - You can use image_search tool if an image would enhance the content

                Generate ONLY the HTML for ONE slide.""")
        
        response = await llm_with_tools.ainvoke(slide_messages)
        
        # Handle image search if tool calls were made
        while hasattr(response, 'tool_calls') and response.tool_calls and image_count < 4:
            slide_messages.append(response)
            
            for tool_call in response.tool_calls:
                image_count += 1
                query = tool_call['args'].get('query', 'business')
                print(f"  📸 Searching for image: '{query}'")
                
                tool_result = image_search.invoke(query)
                slide_messages.append({
                    "role": "tool",
                    "content": tool_result,
                    "tool_call_id": tool_call['id']
                })
            
            response = await llm_with_tools.ainvoke(slide_messages)
        
        slide_html = response.content.strip()
        all_slides_html.append(slide_html)
        print(f"✅ Slide {i} generated")
    
    # Combine all slides
    slides_html = '\n\n'.join(all_slides_html)
    print(f"\n✅ Generated {len(all_slides_html)} slides with {image_count} images")
    
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
