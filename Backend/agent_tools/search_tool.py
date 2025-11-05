from langchain.tools import tool

@tool
def web_search_tool(query: str):
    """ Web tool that will search whole internet for given query and return relevant info """
    