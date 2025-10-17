import os
from langchain_core.tools import tool


@tool
def save_pitch_deck_html(
    slides_html: str,
    filename: str = "pitch_deck.html",
    title: str = "Investment Pitch Deck"
) -> str:
    """
    Save HTML pitch deck slides to a complete presentation file.
    
    Args:
        slides_html: Complete HTML string containing all slide <div> elements
        filename: Filename to save the pitch deck to (default: pitch_deck.html)
        title: Title for the presentation (default: Investment Pitch Deck)
    
    Returns:
        Message indicating the pitch deck was saved successfully.
    """
    try:
        # Ensure output directory exists
        os.makedirs("output", exist_ok=True)
        
        # Validate that we have slide HTML
        if not slides_html or 'class="slide' not in slides_html:
            return "Error: No valid slide HTML provided. Slides must contain elements with class='slide'."
        
        # Read the HTML template
        template_path = os.path.join("templates", "pitch_deck_template.html")
        if not os.path.exists(template_path):
            return f"Error: Template file not found at {template_path}"
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        # Replace placeholders in template
        html_content = template.replace("{{title}}", title)
        html_content = html_content.replace("{{slides}}", slides_html)
        
        # Save the HTML file
        file_path = os.path.join("output", filename)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        # Count slides  
        import re
        slide_pattern = r'<div\s+class="slide"'  # Matches: <div class="slide" or <div class="slide "
        slide_count = len(re.findall(slide_pattern, slides_html))
        
        return f"✅ Pitch deck saved to {file_path} with {slide_count} slides. Open in browser to view."
        
    except Exception as e:
        return f"❌ Error saving HTML pitch deck: {str(e)}"

