import requests
from bs4 import BeautifulSoup
import urllib.parse
import hashlib
import os
from typing import List, Dict, Optional
from langchain_core.tools import tool


# Mapping of business/pitch deck terms to relevant image search terms
IMAGE_CATEGORY_MAPPING = {
    # Business & Finance
    'business': 'business people meeting office',
    'finance': 'financial graphs charts data',
    'financial': 'financial technology banking',
    'investment': 'investment money growth',
    'growth': 'business growth chart upward',
    'revenue': 'financial success profit',
    'profit': 'business success money',
    
    # Market & Strategy
    'market': 'business market strategy',
    'opportunity': 'business opportunity success',
    'strategy': 'business strategy planning',
    'competitive': 'business competition strategy',
    'analysis': 'data analysis business',
    
    # Team & People
    'team': 'business team collaboration',
    'collaboration': 'team working together office',
    'leadership': 'business leader team',
    'people': 'business professionals office',
    
    # Technology & Innovation
    'technology': 'modern technology innovation',
    'innovation': 'innovation technology future',
    'ai': 'artificial intelligence technology',
    'digital': 'digital technology modern',
    'software': 'software development technology',
    'app': 'mobile app technology',
    
    # Problem & Solution
    'problem': 'problem solving business',
    'solution': 'solution innovation success',
    'challenge': 'business challenge overcome',
    
    # Vision & Goals
    'vision': 'business vision future success',
    'goal': 'target goal achievement',
    'success': 'business success achievement',
    'future': 'future innovation technology',
}


def _enhance_search_query(description: str) -> str:
    """Enhance search query based on business context."""
    description_lower = description.lower()
    
    # Check for mapped categories
    for key, enhanced in IMAGE_CATEGORY_MAPPING.items():
        if key in description_lower:
            return enhanced
    
    # Default: add business context
    return f"{description} business professional"


def _search_pexels_api(query: str, limit: int = 1) -> List[Dict[str, str]]:
    """
    Search Pexels using their free API.
    Get API key from: https://www.pexels.com/api/
    Set as environment variable: PEXELS_API_KEY
    """
    api_key = os.getenv('PEXELS_API_KEY')
    if not api_key:
        return []
    
    try:
        url = f"https://api.pexels.com/v1/search"
        headers = {'Authorization': api_key}
        params = {
            'query': query,
            'per_page': limit,
            'orientation': 'landscape'
        }
        
        response = requests.get(url, headers=headers, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = []
            
            for photo in data.get('photos', []):
                results.append({
                    'url': photo['src']['large'],
                    'alt': photo.get('alt', query),
                    'source': 'Pexels API'
                })
            
            return results
    except Exception as e:
        print(f"Pexels API error: {str(e)}")
    
    return []


def _search_pixabay_api(query: str, limit: int = 1) -> List[Dict[str, str]]:
    """
    Search Pixabay using their free API.
    Get API key from: https://pixabay.com/api/docs/
    Set as environment variable: PIXABAY_API_KEY
    """
    api_key = os.getenv('PIXABAY_API_KEY')
    if not api_key:
        return []
    
    try:
        url = "https://pixabay.com/api/"
        params = {
            'key': api_key,
            'q': query,
            'image_type': 'photo',
            'orientation': 'horizontal',
            'per_page': limit,
            'safesearch': 'true'
        }
        
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            results = []
            
            for hit in data.get('hits', []):
                results.append({
                    'url': hit['largeImageURL'],
                    'alt': hit.get('tags', query),
                    'source': 'Pixabay API'
                })
            
            return results
    except Exception as e:
        print(f"Pixabay API error: {str(e)}")
    
    return []


def _get_unsplash_photo(query: str) -> Optional[str]:
    """
    Use Unsplash's open photo collection.
    Returns a direct image URL or None.
    """
    try:
        # Use Unsplash Source (still works for some queries)
        # Format: https://source.unsplash.com/featured/?{query}
        test_url = f"https://source.unsplash.com/1600x900/?{urllib.parse.quote(query)}"
        
        # Test if URL works
        response = requests.head(test_url, timeout=5, allow_redirects=True)
        if response.status_code == 200:
            return response.url  # Return the redirected actual image URL
    except Exception:
        pass
    
    return None


@tool
def image_search(query: str, limit: int = 5) -> str:
    """
    Search for images related to a query with business context.
    
    Args:
        query: Search query for images (e.g., "business growth chart", "team collaboration")
        limit: Maximum number of image results to return (default: 5)
    
    Returns:
        JSON string containing list of image results with URLs and descriptions
    """
    import json
    
    # Enhance query for better business-relevant results
    enhanced_query = _enhance_search_query(query)
    print(f"Image search: '{query}' -> '{enhanced_query}'")
    
    results = []
    
    # Try Pexels API first (best quality, most relevant)
    if not results:
        results = _search_pexels_api(enhanced_query, limit)
    
    # Try Pixabay API
    if not results:
        results = _search_pixabay_api(enhanced_query, limit)
    
    # Try Unsplash
    if not results:
        unsplash_url = _get_unsplash_photo(enhanced_query)
        if unsplash_url:
            results = [{
                'url': unsplash_url,
                'alt': query,
                'source': 'Unsplash'
            }]
    
    # Return results or empty if none found
    return json.dumps({
        'query': query,
        'enhanced_query': enhanced_query,
        'count': len(results),
        'images': results
    }, indent=2)


@tool
def get_stock_image_url(description: str, width: int = 1200, height: int = 675) -> str:
    """
    Get a relevant, working stock image URL based on a description.
    Tries multiple sources to find business-relevant images.
    
    Args:
        description: Description of the image needed
        width: Image width in pixels (default: 1200)
        height: Image height in pixels (default: 675 for 16:9 aspect)
    
    Returns:
        Direct URL to a relevant stock image
    """
    # Enhance query for business relevance
    enhanced_query = _enhance_search_query(description)
    print(f"  Image for '{description}' -> searching '{enhanced_query}'")
    
    # Try Pexels API (free, good quality, relevant)
    try:
        results = _search_pexels_api(enhanced_query, limit=1)
        if results:
            print(f"    ✓ Found via Pexels API")
            return results[0]['url']
    except Exception:
        pass
    
    # Try Pixabay API
    try:
        results = _search_pixabay_api(enhanced_query, limit=1)
        if results:
            print(f"    ✓ Found via Pixabay API")
            return results[0]['url']
    except Exception:
        pass
    
    # Try Unsplash
    try:
        url = _get_unsplash_photo(enhanced_query)
        if url:
            print(f"    ✓ Found via Unsplash")
            return url
    except Exception:
        pass
    
    # Return None if no image sources are available
    # This will signal to use CSS background or no image
    print(f"    ℹ No image API available, will use design elements only")
    return ""

