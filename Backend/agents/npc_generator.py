from dataclasses import dataclass
from PIL import Image
from typing import TypedDict, Annotated, Optional
from uuid import uuid4 
import io
import base64
import torch

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.chat_models import init_chat_model
from deepagents import create_deep_agent

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


from system_prompt.search_agent_prompt import search_prompt
from agent_tools.search_tool import web_search_tool
from system_prompt.query_process_prompt import query_processing
from agent_tools.image_generation import img_pipeline
from system_prompt.negative_prompt import negative_prompt_generation

class AgentState(TypedDict):
    query: str
    key_info: Optional[str] 
    additional_info: Optional[str] 
    explanation: Optional[str] 
    messages: Annotated[list, add_messages] # Stores conversation history

@dataclass
class QueryProcessingResponse:
    user_query : str
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

@dataclass
class ImageGenerationPrompt:
    query_response: str
    search_key_info : str
    search_additional_info : str


llm = init_chat_model(
    model="gpt-5",
)



class NPCBuilder:

    def __init__(self):
        self.llm = llm,
        self.query_processing_agent = create_agent(
                model=llm,
                system_prompt=query_processing,
                response_format=QueryProcessingResponse
            )
        
        self.search_agent = create_agent(
                model=llm,
                tools=[web_search_tool],
                system_prompt=search_prompt,
                response_format=WebSearchResponse
            )
        

        self.image_generation_tool = img_pipeline
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        self.negative_prompt = negative_prompt_generation
        self.height = 512
        self.width = 512
    
    def save_image(self,image:Image,filename:str):
        """ Save generated Image to File """
        try:
            with open(filename, "wb") as f:
                image.save(f, format="PNG")

        except Exception as e:
            raise ValueError(f"Error in save Image: {e}")

    def generate_query_response(self,query:str,state:AgentState):
        """ Generate NPC Character Image from given query """
        try:
            self.query_processing_agent =

            # `thread_id` is a unique identifier for a given conversation.
            config = {"configurable": {"thread_id": "1"}}

            query_response = self.query_processing_agent.invoke(
                {"messages": [{"role": "user", "content": "what is the weather outside?"}]},
                config=config,
                context=Context(user_id=str(uuid4()))
            )

            return query_response
        
        except Exception as e:
            raise ValueError(f"Error in generating query response: {e}")
        
    
    def generate_search_response(self,query_response:QueryProcessingResponse,state:AgentState):
        """ Internet Search to get relevant info about query """

        try:
            self.search_agent = 

            search_response = self.search_agent.invoke(
                {"messages": [{"role": "user", "content": query_response.structured_summary}]},
                context=Context(user_id=str(uuid4()))
            )

            return search_response
        
        except Exception as e:
            raise ValueError(f"Error in generating search response: {e}")
    
    def generate_img_prompt(
            self,
            search_response:WebSearchResponse,
            query_response: QueryProcessingResponse,
            state:AgentState
            ):
        
        """ Generate Enchanced Image Prompt from Search response and Query Response """

        try:
            self.img_prompt_processing_agent = create_agent(
                model= llm,
                system_prompt=query_response + search_response.key_info + search_response.additional_info + search_response.explanation,
                response_format=str,
            )

            img_prompt = self.img_prompt_processing_agent.invoke(
                {"messages": [{"role": "user", "content": "Generate detailed image prompt based on above information."}]},
                context=Context(user_id=str(uuid4()))
            )

            return img_prompt
        
        except Exception as e:
            raise ValueError(f"Error in generating Image Prompt: {e}")
    
    def generate_npc_image(self,image_prompt:str,state:AgentState):
        """ Generate NPC Image from given Query """
        
        self.img_gen_agent = create_agent(
            tools=[self.image_generation_tool],
            system_prompt=image_prompt,
            response_format=Image.Image,            
        )


    def define_workflow(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("Generate Query Response", self.generate_query_response)
        workflow.add_node("Generate Search Response", self.generate_search_response)
        workflow.add_node("Generate Image Prompt", self.generate_img_prompt)
        workflow.add_node("Generate NPC Image", self.generate_npc_image)

        workflow.add_edge(
            "Generate Query Response",
            "Generate Search Response",
            "Generate Image Prompt",
            "Generate NPC Image"
        )

        workflow.set_entry_point("Generate Query Response")
        workflow.set_finish_point("Generate NPC Image",END)

        main_agent = workflow.compile()
        initial_state = {"messages": [("user", "Start the Agent Flow.")]}
        result = main_agent.invoke(initial_state)

        return result

