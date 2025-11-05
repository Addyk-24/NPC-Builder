def search_prompt(query: str) -> str:
    prompt = f"""
You are a search agent. Your task is to generate relevant information based on the user's query {query} from whole internet.
You should provide concise and accurate information that directly addresses the query.

"""
    return prompt