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
        model_name="llama-3.3-70b-versatile",
        temperature=0.7,
        max_tokens=4000
    )
    
    # Bind image_search tool to the LLM for autonomous tool calling
    llm_with_tools = llm.bind_tools([image_search])
    
    print("Content Received: ", "\nDescription: ", state.product_description, "\nMarket Analysis: ", state.market_synthesis, "\nInnovation: ", state.current_proposal)
    
    # Step 1: Generate a high-level plan
    print("\n\n Step 1: Creating slide structure plan...")
    plan_messages = [
        SystemMessage(content="""Create a detailed plan for a pitch deck with 10-20 slides. For each slide, specify:
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
    print(" Plan created")
    print(f"\n SLIDE PLAN:\n{slide_plan}\n")
    
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
            SystemMessage(content="""Generate ONE slide using Tailwind CSS. Use this basic format:

            <div class="slide bg-gradient-to-br from-blue-50 to-indigo-100 p-16 min-h-screen flex gap-8">
                <div class="flex-1 flex flex-col justify-center">
                    <h2 class="text-5xl font-bold text-gray-900 mb-8 font-serif flex items-center gap-4">
                        <i class="fas fa-chart-line text-blue-600"></i>Title
                    </h2>
                    <div class="space-y-6 text-lg leading-relaxed text-gray-700">
                        <!-- Content here -->
                    </div>
                </div>
                <div class="flex-1 flex items-center justify-center">
                    <img src="image-url" class="w-96 h-72 object-cover rounded-xl shadow-2xl" alt="Description">
                </div>
                <div class="slide-number">1</div>
            </div>

            CRITICAL: Return ONLY the HTML div element, NO markdown blocks, NO backticks, NO function calls.

            LAYOUT OPTIONS:
            - Side-by-side: flex with two flex-1 divs (text left/right, image right/left - vary the order)
            - Centered: flex-col items-center justify-center text-center (for title slides)
            - Image-above: flex-col with image at top, text below
            - Grid: grid grid-cols-2 gap-8 (balanced layout)
            - Custom: Create your own creative layouts using flex, grid, or other Tailwind utilities

            STYLING:
            - Backgrounds: bg-gradient-to-br from-blue-50 to-indigo-100, from-slate-900 to-gray-800, from-purple-50 to-pink-100, bg-white
            - Images: w-96 h-72, w-80 h-60, w-full h-80, w-full max-w-2xl h-80
            - Icons: fas fa-chart-line, fas fa-users, fas fa-star, fas fa-lightbulb with colors text-blue-600, text-purple-600, text-yellow-400
            - Stats: Use colored boxes with text-3xl font-bold and colors text-green-600, text-blue-600, text-red-600

            AVAILABLE CDNs (Use the below tools to each and every slide to improve the look and feel):
            - Fonts: font-sans (Inter), font-serif (Playfair Display), font-poppins (Poppins), font-montserrat (Montserrat), font-roboto (Roboto)
            - Icons: Font Awesome (fas fa-*), Bootstrap Icons (bi bi-*), Feather Icons (feather-*)
            - Animations: animate.css classes (animate-fadeIn, animate-slideInUp, animate-bounce), AOS (data-aos="fade-up")
            - Charts: Chart.js for data visualization
            - 3D Effects: Three.js for 3D elements, Particles.js for particle effects
            - Advanced Animations: GSAP for smooth animations

            TOOL USAGE:
            - ALWAYS call image_search tool for each slide if image is needed
            - Use queries like "business meeting", "growth chart", "technology", "perfume dispenser"
            - Use returned image URLs in <img> tags with proper styling
            - Include proper alt text for accessibility

            CREATIVE FREEDOM:
            - VARY your layouts and backgrounds for visual interest
            - Experiment with different arrangements (image left/right, top/bottom)
            - Use creative positioning and sizing for images
            - Try different background gradients and color schemes
            - Mix and match layout elements creatively

            Write 150-250 words with metrics and data."""),
                                    HumanMessage(content=f"""Generate slide {i} with title: "{slide_title}"

                        Based on this information:
                        PRODUCT: {state.product_description}
                        MARKET ANALYSIS: {state.market_synthesis}
                        INNOVATION: {state.current_proposal}

                        Slide content points from plan:
                        {section}

                        Write detailed, comprehensive content for this single slide.""")
        ]
        
        
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
                    "content": f"Image search result: {tool_result}. Use the image URL from this result in your HTML <img> tag with proper styling: class='w-96 h-72 object-cover rounded-xl shadow-2xl'",
                    "tool_call_id": tool_call['id']
                })
            
            response = await llm_with_tools.ainvoke(slide_messages)
        
        slide_html = response.content.strip()
        
        # Clean up any markdown code blocks that might have been generated
        if slide_html.startswith('```html'):
            slide_html = slide_html.replace('```html', '').replace('```', '').strip()
        elif slide_html.startswith('```'):
            slide_html = slide_html.replace('```', '').strip()
        
        # Clean up function calls and replace with placeholder images
        import re
        slide_html = re.sub(r'<function=image_search>.*?</function>', 
                           'https://images.pexels.com/photos/10600142/pexels-photo-10600142.jpeg?auto=compress&cs=tinysrgb&h=650&w=940', 
                           slide_html)
        slide_html = re.sub(r'<img src=json\.loads\(<function=image_search>.*?</function>\)\[.*?\]', 
                           '<img src="https://images.pexels.com/photos/10600142/pexels-photo-10600142.jpeg?auto=compress&cs=tinysrgb&h=650&w=940"', 
                           slide_html)
        
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
