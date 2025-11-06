from dataclasses import dataclass

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from langgraph.graph import StateGraph

from system_prompt.search_agent_prompt import search_prompt
from agent_tools.search_tool import web_search_tool
from system_prompt.query_process_prompt import query_processing


from langchain_huggingface.llms import HuggingFacePipeline
from transformers import pipeline
from diffusers import DiffusionPipeline


@dataclass
class QueryProcessingResponse:
    user_query = str
    structured_summary: str

@dataclass
class Context:
    """Custom runtime context schema."""
    user_id: str

@dataclass
class WebSearchResponse:
    query_response: QueryProcessingResponse
    key_info: str
    additional_info: str
    explanation: str


model_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")


hf = HuggingFacePipeline(pipeline=model_pipe)

llm = init_chat_model(
    model="gpt-5",
)

image_llm = ""



class NPCBuilder:

    def __init__(self):
        self.llm = llm,
        self.query_processing_agent = None
        self.search_agent = None
        self.img_prompt_processing_agent = None
        self.img_gen_agent = None
    
    def generate_query_response(self,query:str):
        """ Generate NPC Character Image from given query """
        self.query_processing_agent = create_agent(
            model=llm,
            system_prompt=query_processing,
            response_format=QueryProcessingResponse()
        )

        # `thread_id` is a unique identifier for a given conversation.
        config = {"configurable": {"thread_id": "1"}}

        query_response = self.query_processing_agent.invoke(
            {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
            config=config,
            context=Context(user_id="1")
        )

        return query_response
    
    def generate_search_response(self,query_response:QueryProcessingResponse):
        """ Internet Search to get relevant info about query """

        self.search_agent = create_agent(
            model=llm,
            tools=[web_search_tool],
            system_prompt=search_prompt,
            response_format=WebSearchResponse()
        )

        search_response = self.search_agent.invoke(
            {"messages": [{"role": "user", "content": query_response.structured_summary}]},
            context=Context(user_id="1")
        )

        return search_response
    
    def generate_img_prompt(self,search_response:WebSearchResponse):
        """ Generate Enchanced Image Prompt from Search response and Query Response """

        self.img_prompt_processing_agent = create_agent(
            model= llm,
            system_prompt=search_response.key_info + search_response.additional_info,
            response_format=str,
        )

        img_prompt = self.img_prompt_processing_agent.invoke(
            {"messages": [{"role": "user", "content": "Generate detailed image prompt based on above information."}]},
            context=Context(user_id="1")
        )

        return img_prompt
    
    def generate_npc_image(self,image_prompt:str):
        """ Generate NPC Image from given Query """
        
        self.img_gen_agent = create_agent(
            model=image_llm,
            system_prompt=image_prompt,
            response_format=bytes,            
        )

    def create_npc_agent(self):
        """ Generating NPC From Subagents """

        subagents = [self.query_processing_agent,self.search_agent,self.img_prompt_processing_agent,self.img_gen_agent]

        npc_agent = create_deep_agent(
            model = llm,
            instructions=""" You are NPC Agent that takes query as input and generate image of that given input and give that image as output 
            Important:
            - Be more precise on query like what query really tell and then generate image
            """,
            description=""" You have Generated lot of image from given query and now you are improving it and stick to point like generate image what query has stated """,
            tools=[],
            subagents=subagents,
            
        )
