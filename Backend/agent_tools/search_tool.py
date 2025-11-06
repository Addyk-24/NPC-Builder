from langchain.tools import tool
from langchain_community.tools import BraveSearch

tool = BraveSearch()

@tool
def web_search_tool(query: str):
    """ Web tool that will search whole internet for given query and return relevant info """

    search_results = tool.run(query)

    return search_results
