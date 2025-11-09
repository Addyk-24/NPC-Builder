from dotenv import load_dotenv
load_dotenv()
import os, tempfile, gc
from dataclasses import dataclass
from PIL import Image
from typing import TypedDict, Annotated, Optional
from uuid import uuid4 
import io
import base64
import torch

from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_groq import ChatGroq

from deepagents import create_deep_agent

from diffusers import DiffusionPipeline

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages


from agent_tools.search_tool import web_search_tool
from system_prompt.query_process_prompt import query_processing
from system_prompt.search_agent_prompt import search_prompt
from system_prompt.negative_prompt import negative_prompt_generation

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

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


llm = ChatGroq(
    model="llama-3.1-8b-instant",

)

# model_path = "stabilityai/sd-turbo"
model_path = model_path = "runwayml/stable-diffusion-v1-5" 


class NPCBuilder:

    def __init__(self):
        self.llm = llm,
        # Load Diffusion Image Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img_pipe = DiffusionPipeline.from_pretrained(
            model_path,
        ).to(self.device)
            # torch_dtype=torch.float16

        # System Prompts
        self.negative_prompt = negative_prompt_generation()
        self.query_response = query_processing()
        self.search_response = search_prompt()

        # Agents
        self.query_processing_agent = create_agent(
                model=llm,
                system_prompt=self.query_response,
                response_format=QueryProcessingResponse
            )
        
        self.search_agent = create_agent(
                model=llm,
                tools=[web_search_tool],
                system_prompt=self.search_response,
                response_format=WebSearchResponse
            )

        self.num_inference_steps = 25
        self.guidance_scale =7.5
        self.height = 512
        self.width = 512

        self.workflow = None
        self.workflow_agent = None
    
    def save_image(self,image:base64,filename:str):
        """ Save generated Image to File """
        try:
                image.save(filename, format="PNG")

        except Exception as e:
            raise ValueError(f"Error in save Image: {e}")

    # Node 1
    def generate_query_response(self,state:AgentState) -> AgentState:
        """ Generate NPC Character Image from given query """
        try:

            config = {"configurable": {"thread_id": str(uuid4())}}

            response_dict  = self.query_processing_agent.invoke(
                {"messages": [{"role": "user", "content": state["query"]}]},
                config=config,
                )
            
            query_response = response_dict["messages"][-1].content
            state["structured_summary"] = query_response
            state["messages"].append({"role" : "assistant","content" : f"Processed Query: {query_response}"})

            return state
        
        except Exception as e:
            state["messages"].append({"role": "assistant", "content": f"Error processing query: {e}"})
            return state
                
    # Node 2
    def generate_search_response(self,state:AgentState) -> AgentState:
        """ Internet Search to get relevant info about query """

        try:
            
            config = {"configurable": {"thread_id": str(uuid4())}}

            response_dict = self.search_agent.invoke(
                {"messages": [{"role": "user", "content": state["structured_summary"]}]},
                config = config
            )

            search_response = response_dict["messages"][-1].content

            state["key_info"] = search_response
            state["additional_info"] = ""
            state["explanation"] = ""
            state["messages"].append({"role":"assistant","content":f"Processed Web Search Completed :) "})

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


            Create a detailed Stable Diffusion prompt for an NPC character image.
            Include: visual appearance, clothing, pose, lighting, art style, quality modifiers.
            Keep it under 77 tokens.
            
            """
            self.img_prompt_processing_agent = create_agent(
                model= llm,
                system_prompt="You are an expert at creating Stable Diffusion prompts. Generate concise, visual, detailed prompts."
            )

            img_prompt = self.img_prompt_processing_agent.invoke(
                {"messages": [{"role": "user", "content": context}]},
                config= config
            )

            state["image_prompt"] = img_prompt["messages"][-1].content

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
            ).images[0]

            # Save locally
            filename = f"npc_{uuid4()}.png"
            npc_image.save(filename)

            tmp_dir = tempfile.gettempdir()
            filename = os.path.join(tmp_dir, f"npc_{uuid4()}.png")
            npc_image.save(filename)

            # Now Converting to base64
            buffered = io.BytesIO()

            npc_image.save(buffered, format = "PNG")

            img_str = base64.b64encode(buffered.getvalue()).decode()

            state["generated_image"] = img_str

            state["messages"].append({"role": "assistant", "content": f"Image generated and saved as {filename}"})

            # # Cleanup after done
            # del image
            # torch.cuda.empty_cache()
            # gc.collect()

            return state
        except Exception as e:
            state["messages"].append({"role": "assistant", "content": f"Error generating image: {e}"})
            return state


    def build_workflow(self):
        """ Define and Build WOrkflow for Sequential Flow of Agent """

        self.workflow = StateGraph(AgentState)

        self.workflow.add_node("Generate Query Response", self.generate_query_response)
        self.workflow.add_node("Generate Search Response", self.generate_search_response)
        self.workflow.add_node("Generate Image Prompt", self.generate_img_prompt)
        self.workflow.add_node("Generate NPC Image", self.generate_npc_image)

        self.workflow.add_edge("Generate Query Response","Generate Search Response")
        self.workflow.add_edge("Generate Search Response","Generate Image Prompt")
        self.workflow.add_edge("Generate Image Prompt","Generate NPC Image")
        self.workflow.add_edge("Generate NPC Image", END)

        self.workflow.set_entry_point("Generate Query Response")

        main_agent = self.workflow.compile()
        self.workflow_agent = main_agent

    def run_agent(self,query:str):
        """ Execute NPC Character Full Generation Pipeline """

        if self.workflow is None:
            self.build_workflow()

        initial_state = {
            "query" : query,
            "messages" : []
        }
        result = self.workflow_agent.invoke(initial_state)        

        print("Message: ",result["messages"])
        print("NPC AGENT DONEEE!!! ")

        return result["messages"]

if __name__ == "__main__":
    builder = NPCBuilder()
    # prompt ="Create a npc character which is indian origin and wearing indian cops uniform"
    prompt = """
Official Ken Sugimori style Pokémon art of a creature. A Ground and Steel type monster. It has the body shape of a large, armored armadillo, with heavy metal plates, a drill on its nose, and glowing red eyes. Its color palette is brown, grey, and orange. Dynamic pose, simple background, smooth, clean line art, high detail, trending on ArtStation.

"""

    agent_result = builder.run_agent(prompt)

    # print(agent_result)



# """
#    Character concept art, a detailed NPC character design in the style of Fortnite, vibrant colors, stylized 3D render, cel-shaded, Unreal Engine, trending on ArtStation. The character is a [description of NPC role, e.g., 'rugged space bounty hunter'], [description of physical traits, e.g., 'tall, muscular male'], wearing [description of outfit and accessories, Full body shot, dynamic pose, clean background.
# """

# """
# Official Ken Sugimori style Pokémon art of a creature. A Ground and Steel type monster. It has the body shape of a large, armored armadillo, with heavy metal plates, a drill on its nose, and glowing red eyes. Its color palette is brown, grey, and orange. Dynamic pose, simple background, smooth, clean line art, high detail, trending on ArtStation.

# """