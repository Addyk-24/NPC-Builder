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

from diffusers import DiffusionPipeline

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


from system_prompt.search_agent_prompt import search_prompt
from agent_tools.search_tool import web_search_tool
from system_prompt.query_process_prompt import query_processing
from agent_tools.image_generation import img_pipeline
from system_prompt.negative_prompt import negative_prompt_generation

class AgentState(TypedDict):
    query: str
    structured_summary: Optional[str]
    key_info: Optional[str] 
    additional_info: Optional[str] 
    explanation: Optional[str]
    image_prompt: Optional[str]
    generated_image: Optional[str]  # base64 encoded
    messages: Annotated[list, add_messages]

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
        # Load Diffusion Image Model
        self.device = "cude" if torch.cude.is_available() else "cpu"
        self.img_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

        self.image_generation_tool = img_pipeline
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        self.negative_prompt = negative_prompt_generation
        self.height = 512
        self.width = 512

        self.workflow = None
    
    def save_image(self,image:base64,filename:str):
        """ Save generated Image to File """
        try:
                image.save(filename, format="PNG")

        except Exception as e:
            raise ValueError(f"Error in save Image: {e}")

    # Node 1
    def generate_query_response(self,query:str,state:AgentState) -> AgentState:
        """ Generate NPC Character Image from given query """
        try:

            # `thread_id` is a unique identifier for a given conversation.
            config = {"configurable": {"thread_id": str(uuid4())}}

            query_response = self.query_processing_agent.invoke(
                {"messages": [{"role": "user", "content":state["query"]}]},
                config=config,
                context=Context(user_id=str(uuid4()))
            )

            state["structured_summary"] = query_response.structured_summary
            state["messages"].append({"role" : "NPC-Agent","content" : f"Processed Query: {query_response.structured_summary}"})

            return state
        
        except Exception as e:
            state["messages"].append({"role": "assistant", "content": f"Error processing query: {e}"})
            return state
                
    # Node 2
    def generate_search_response(self,state:AgentState) -> AgentState:
        """ Internet Search to get relevant info about query """

        try:
            
            config = {"configurable": {"thread_id": str(uuid4())}}

            search_response = self.search_agent.invoke(
                {"messages": [{"role": "user", "content": state["structured_summary"]}]},
                config = config
            )

            state["key_info"] = search_response.key_info
            state["additional_info"] = search_response.additional_info
            state["explanation"] = search_response.explanation
            state["messages"] = [{"role":"NPC-Agent","content":f"Processed Web Search Completed :) "}]

            return state
        
        except Exception as e:
            state["messages"].append({"role": "assistant", "content": f"Error in web search: {e}"})
            return state
        
    
    # Node 3
    def generate_img_prompt(
            self,
            state:AgentState
            ) -> AgentState :
        
        """ Generate Enchanced Image Prompt from Search response and Query Response """

        try:

            config = {"configurable": {"thread_id": str(uuid4())}}

            context = f"""

            Original Query : {state['query']}
            Key Information : {state['key_info']}
            Additional Information : {state['additional_info']}
            Explanation : {state['explanation']}

            Create a detailed Stable Diffusion prompt for an NPC character image.
            Include: visual appearance, clothing, pose, lighting, art style, quality modifiers.
            Keep it under 77 tokens.
            
            """
            self.img_prompt_processing_agent = create_agent(
                model= llm,
                system_prompt="You are an expert at creating Stable Diffusion prompts. Generate concise, visual, detailed prompts."
            )

            img_prompt = self.img_prompt_processing_agent.invoke(
                {"messages": [{"role": "user", "content": {context}}]},
                config= config
            )

            state["image_prompt"] = img_prompt["message"][-1].content
            state["messages"].append({"role": "assistant", "content": f"Image prompt: {state['image_prompt']}"})


            return state
        
        except Exception as e:
            state["messages"].append({"role": "assistant", "content": f"Error creating prompt: {e}"})
            return state
    
    # Node 4
    def generate_npc_image(self,state:AgentState) -> AgentState:
        """ Generate NPC Image from given Query """

        try:
            npc_image = self.img_pipe(
                prompt = state['image_prompt'],
                negative_prompt = self.negative_prompt,
                num_inference_steps = self.num_inference_steps,
                guidance_scale = 7.5,
                height= self.height,
                width = self.width
            ).image[0]

            # Now Converting to base64
            buffered = io.BytesIO()

            npc_image.save(buffered, format = "PNG")

            img_str = base64.b64encode(buffered.getvalue()).decode()

            state["generated_image"] = img_str

            # Saving image generated
            filename = f"npc_{uuid4()}.png"

            self.save_image(npc_image,filename=filename)

            state["messages"].append({"role": "assistant", "content": f"Image generated and saved as {filename}"})

            return state
        except Exception as e:
            state["messages"].append({"role": "assistant", "content": f"Error generating image: {e}"})
            return state


    def build_workflow(self):
        """ Define and Build WOrkflow for Sequential Flow of Agent """

        workflow = StateGraph(AgentState)

        workflow.add_node("Generate Query Response", self.generate_query_response)
        workflow.add_node("Generate Search Response", self.generate_search_response)
        workflow.add_node("Generate Image Prompt", self.generate_img_prompt)
        workflow.add_node("Generate NPC Image", self.generate_npc_image)

        workflow.add_edge("Generate Query Response","Generate Search Response")
        workflow.add_edge("Generate Search Response","Generate Image Prompt")
        workflow.add_edge("Generate Image Prompt","Generate NPC Image")
        workflow.add_edge("Generate NPC Image", END)

        workflow.set_entry_point("Generate Query Response")

        main_agent = workflow.compile()

    def run_agent(self,query:str):
        """ Execute NPC Character Full Generation Pipeline """

        if self.workflow is None:
            self.build_workflow()

        initial_state = {
            "query" : {query},
            "messages" : []
        }
        result = self.workflow.invoke(initial_state)

        return result

if __name__ == "__main__":
    builder = NPCBuilder()
    result = builder.run_agent("Create a cyberpunk street samurai with neon tattoos")
    print(result["messages"])
