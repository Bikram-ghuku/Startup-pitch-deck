"""
Image search tool using Pexels API (FREE).
Get your free API key at: https://www.pexels.com/api/
"""
import requests
import os
from typing import List, Dict
from langchain_core.tools import tool


def _enhance_query(description: str) -> str:
    """Add business context to search query."""
    desc_lower = description.lower()
    print(f"Enhanced query: {desc_lower}")
    return desc_lower

def _search_pexels(query: str, limit: int = 1) -> List[Dict[str, str]]:
    """Search Pexels API for images."""
    api_key = os.getenv('PEXELS_API_KEY')
    if not api_key:
        return []
    
    try:
        response = requests.get(
            "https://api.pexels.com/v1/search",
            headers={'Authorization': api_key},
            params={
                'query': query,
                'per_page': limit,
                'orientation': 'landscape',
                'size': 'large'
            },
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            return [
                {
                    'url': photo['src']['large'],
                    'alt': photo.get('alt', query),
                    'source': 'Pexels'
                }
                for photo in data.get('photos', [])
            ]
    except Exception:
        pass
    
    return []


@tool
def image_search(query: str, limit: int = 5) -> str:
    """
    Search for professional images using Pexels API.
    
    Use SHORT, SPECIFIC queries (2-4 words maximum) for best results:
    - Good: "ai technology", "growth chart", "team meeting", "mobile app"
    - Bad: "business people working in modern office building"
    
    Returns JSON with image URLs that can be embedded in HTML.
    Example: Extract url with: json.loads(result)['images'][0]['url']
    """
    import json
    
    enhanced_query = _enhance_query(query)
    results = _search_pexels(enhanced_query, limit)
    
    return json.dumps({
        'query': query,
        'count': len(results),
        'images': results
    }, indent=2)


@tool
def get_stock_image_url(description: str, width: int = 1200, height: int = 675) -> str:
    """
    Get a stock image URL from Pexels (FREE).
    
    Setup: Set PEXELS_API_KEY environment variable
    Get key: https://www.pexels.com/api/
    
    Returns: Image URL or empty string
    """
    enhanced_query = _enhance_query(description)
    results = _search_pexels(enhanced_query, limit=1)
    
    return results[0]['url'] if results else ""
