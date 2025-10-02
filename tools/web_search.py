import requests
from bs4 import BeautifulSoup
import urllib.parse
from typing import Dict, List
from langchain_core.tools import tool


def _scrape_duckduckgo(query: str) -> List[Dict[str, str]]:
    """Scrape DuckDuckGo HTML results."""
    url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    results = []
    for result in soup.select('div.result')[:5]:
        title_elem = result.select_one('a.result__a')
        snippet_elem = result.select_one('a.result__snippet')
        
        if title_elem:
            results.append({
                'title': title_elem.get_text(strip=True),
                'snippet': snippet_elem.get_text(strip=True) if snippet_elem else '',
                'url': title_elem.get('href', '')
            })
    
    return results


def _format_results(results: List[Dict[str, str]]) -> str:
    """Format search results into readable text."""
    if not results:
        return "No results found."
    
    formatted = []
    for i, result in enumerate(results, 1):
        formatted.append(f"{i}. {result['title']}")
        if result['snippet']:
            formatted.append(f"   {result['snippet']}")
        formatted.append("")  # Empty line between results
    
    return "\n".join(formatted)


@tool
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.
    
    Args:
        query: Search query string
        
    Returns:
        Formatted search results with titles and snippets
    """
    try:
        results = _scrape_duckduckgo(query)
        return _format_results(results)
        
    except requests.Timeout:
        return f"Search timeout for '{query}'. Try a more specific query."
    
    except requests.RequestException:
        return f"Search error: Unable to connect. The query '{query}' could not be searched."
    
    except Exception as e:
        print(f"Unexpected error in web_search: {str(e)}")
        return f"Search failed for '{query}'. Consider this may be a novel or emerging concept."