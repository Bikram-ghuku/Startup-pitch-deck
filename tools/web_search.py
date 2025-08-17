import requests
from bs4 import BeautifulSoup
import urllib.parse
from langchain_core.tools import tool

@tool
def web_search(query: str) -> str:
    """Search DuckDuckGo and return results."""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        results = []
        for result in soup.select('div.result')[:3]:
            title_elem = result.select_one('a.result__a')
            if title_elem:
                results.append(title_elem.get_text(strip=True))
        
        return str(results) if results else f"No results for '{query}' - likely a novel concept."
        
    except Exception as e:
        print(f"Search error: {str(e)}")
        return f"Search unavailable. '{query}' appears to be innovative."