from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from system_prompt.search_agent_prompt import search_prompt
from agent_tools.search_tool import web_search_tool


from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
from diffusers import DiffusionPipeline



model_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")


hf = HuggingFacePipeline(pipeline=model_pipe)

llm = init_chat_model(
    model="gpt-5",
)

image_llm = ""

search_agent = create_agent(
    model=llm,
    tools=[web_search_tool],
    system_prompt=search_prompt,

)

img_agent = create_agent(
    model=llm
    
)


npc_agent = create_deep_agent(
    model = llm,
    instructions=""" You are NPC Agent that takes query as input and generate image of that given input and give that image as output 
    Important:
    - Be more precise on query like what query really tell and then generate image
    """,
    description=""" You have Generated lot of image from given query and now you are improving it and stick to point like generate image what query has stated """,
    tools=[],

)