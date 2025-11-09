from dotenv import load_dotenv
load_dotenv()
import os
from langchain.tools import tool
from langchain_community.utilities import SerpAPIWrapper

SERPAPI_API_KEY = os.environ.get("SERPAPI_API_KEY")

if not SERPAPI_API_KEY:
    raise ValueError("SERPAPI_API_KEY not found in environment variables")

params = {
    "engine": "google",
    "num": 5, 
    "hl": "en",  
}

search_wrapper = SerpAPIWrapper(params=params)

@tool
def web_search_tool(query: str) -> str:
    """ Web tool that will search whole internet for given query and return relevant info """

    try:
        search_results = search_wrapper.run(query)
        return search_results
    
    except Exception as e:
        raise ValueError(f" :( Search temporarily unavailable. Error: {e} ")
