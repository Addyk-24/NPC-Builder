from dataclasses import dataclass
from PIL import Image
from typing import TypedDict, Annotated, Optional

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
    query_response: QueryProcessingResponse
    search_key_info : str
    search_additional_info : str


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
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        self.negative_prompt = negative_prompt_generation
        self.height = 512
        self.width = 512
    
    def generate_query_response(self,query:str,state:AgentState):
        """ Generate NPC Character Image from given query """
        try:
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
        
        except Exception as e:
            raise ValueError(f"Error in generating query response: {e}")
        
    
    def generate_search_response(self,query_response:QueryProcessingResponse,state:AgentState):
        """ Internet Search to get relevant info about query """

        try:
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
                context=Context(user_id="1")
            )

            return img_prompt
        
        except Exception as e:
            raise ValueError(f"Error in generating Image Prompt: {e}")
    
    def generate_npc_image(self,image_prompt:str,state:AgentState):
        """ Generate NPC Image from given Query """
        
        self.img_gen_agent = create_agent(
            model=image_llm,
            tools=[img_pipeline],
            system_prompt=image_prompt,
            response_format=bytes,            
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
