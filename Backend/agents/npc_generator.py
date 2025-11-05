from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

from system_prompt.search_agent_prompt import search_prompt


llm = HuggingFace(
        id="Intel/sd-1.5-square-quantized",
        max_tokens=4096,
    ),

search_agent = create_agent(
    model=llm,
    

)

npc_agent = Agent(
    model = llm,
    instructions=""" You are NPC Agent that takes query as input and generate image of that given input and give that image as output 
    Important:
    - Be more precise on query like what query really tell and then generate image
    """,
    description=""" You have Generated lot of image from given query and now you are improving it and stick to point like generate image what query has stated """,
    tools=[],

)