def search_prompt(query: str) -> str:
    prompt = f"""
You are a search agent. Your task is to generate relevant information based on the user's query {query} from whole internet.
You should provide concise and accurate information that directly addresses the query.
## FORMAT OF RESPONSE:
Key Information: <Relevant information based on the query>
Additional Details: <Any additional context or details that might be useful>
Explanation: <Brief explanation of how the information was gathered and its relevance to the query>


"""
    return prompt