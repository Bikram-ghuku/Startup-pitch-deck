"""
Web search tool using DuckDuckGo via langchain_community.
"""
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import tool

# Create the DuckDuckGo search instance
_ddg_search = DuckDuckGoSearchRun()


@tool
def web_search(query: str) -> str:
    """
    Search the web using DuckDuckGo and return formatted results.
    
    Args:
        query: Search query string
        
    Returns:
        Search results from DuckDuckGo
    """
    try:
        results = _ddg_search.invoke(query)
        return results
    except Exception as e:
        print(f"Error in web_search: {str(e)}")
        return f"Search failed for '{query}'. Error: {str(e)}"
