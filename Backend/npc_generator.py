from dotenv import load_dotenv
load_dotenv()
import os, tempfile, gc
import numpy as np
from dataclasses import dataclass
from PIL import Image
from typing import TypedDict, Annotated, Optional
from uuid import uuid4 
import io
import base64
import torch

import open3d as o3d


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
from agent_tools.image_transformation import generate_depth_scale, normal_map_from_depth

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class AgentState(TypedDict):
    # Query processing
    query: str
    structured_summary: Optional[str]
    key_info: Optional[str] 
    additional_info: Optional[str] 
    explanation: Optional[str]

    # Img gen
    image_prompt: Optional[str]
    generated_image: Optional[str]  # base64 encoded

    # Normal mapping
    depth_array : Optional[np.ndarray]
    normal_map : Optional[np.ndarray]
    normal_map_path : Optional[str]
    mesh_file : Optional[str]
    mesh_file : Optional[str]

    # Post Task Messages
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

        # Agent tools
        self.search_response = search_prompt()
        self.depth_scale_generation = generate_depth_scale()
        self.normal_map_generation = normal_map_from_depth()

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
            Keep it under 90 tokens.
            
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

    def generate_normal_map(self,image_path:str):
        """Node 5: Generate normal map from depth scale map"""
        try:

            depth_array = generate_depth_scale(image_path)

            self.state["depth_array"] = depth_array

            if depth_array is None:
                self.state["message"].append({
                    "role":"assistant",
                    "content": "No map Found, skipping Normal map gen"
                })
                return self.state
            
            normal_map = normal_map_from_depth(depth_array,depth_scale=2.0)

            normal_img = Image.fromarray(normal_map)
            normal_filename = f"normal_{uuid4()}.png"
            normal_img.save(normal_filename)
            
            self.state["normal_map"] = normal_map                    
            self.state["normal_map_path"] = normal_filename

            self.state["messages"].append({
                "role": "assistant",
                "content": f"Normal map generated: {normal_filename}"
            })

            return self.state
        
        except Exception as e:
            self.state["messages"].append({
                "role": "assistant",
                "content": f"Error generating normal map: {e}"
            })
            return self.state

    def generate_point_cloud(self, state: AgentState) -> AgentState:
        """ Node 6: Generation of 3D Point Cloud from normal map"""
        try:
            depth_array = state["depth_array"]
            img_base64 = state["generated_image"]
            img_data = base64.b64decode(img_base64)
            npc_image = np.array(Image.open(io.BytesIO(img_data)))

            # Camera intrinsics
            height, width = depth_array.shape
            fx = fy = 500
            cx, cy = width / 2, height / 2

            points = []
            colors = []

            for v in range(0,height,2):
                for u in range(0,width,2):
                    z= depth_array[u,v] * 100
                    if z >0:
                        x = (u-cx) * z / fx
                        y = (v-cy) * z / fy
                        points.append([x, y, z])
                        colors.append(npc_image[v, u] / 255.0)
                    
                pcd  = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.array(points))
                pcd.colors = o3d.utility.Vector3dVector(np.array(colors))

                # Estimating normals if not proceeding with normal map
                pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=30))
                
                # Save point cloud
                pcd_filename = f"pointcloud_{uuid4()}.ply"
                o3d.io.write_point_cloud(pcd_filename, pcd)

                state["point_cloud"] = pcd
                state["point_cloud_path"] = pcd_filename
                
                state["messages"].append({
                    "role": "assistant",
                    "content": f"Point cloud created: {pcd_filename}"
                })
                
                return state

        except Exception as e:
            state["messages"].append({
                "role": "assistant",
                "content": f"Error creating point cloud: {e}"
            })
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
A young female samurai disguised as a traveling monk, her face calm but eyes filled with vengeance. Wears layered monk robes over light armor, carrying a katana wrapped in cloth. Snow falls around her as she walks through a desolate mountain temple. Cinematic lighting, detailed environment, Japanese mythology atmosphere, emotional realism.
"""

    agent_result = builder.run_agent(prompt)

    # print(agent_result)



# """
#    Character concept art, a detailed NPC character design in the style of Fortnite, vibrant colors, stylized 3D render, cel-shaded, Unreal Engine, trending on ArtStation. The character is a [description of NPC role, e.g., 'rugged space bounty hunter'], [description of physical traits, e.g., 'tall, muscular male'], wearing [description of outfit and accessories, Full body shot, dynamic pose, clean background.
# """

# """
# Official Ken Sugimori style Pokémon art of a creature. A Ground and Steel type monster. It has the body shape of a large, armored armadillo, with heavy metal plates, a drill on its nose, and glowing red eyes. Its color palette is brown, grey, and orange. Dynamic pose, simple background, smooth, clean line art, high detail, trending on ArtStation.

# """