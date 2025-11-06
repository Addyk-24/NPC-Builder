def query_processing(query: str) -> str:
    prompt = f"""
Given the user query: {query}, analyze and break down the query to understand its intent and key components. 
Provide a structured summary that highlights the main objectives and any specific details that need to be addressed.

"""
    return prompt