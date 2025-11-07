from diffusers import DiffusionPipeline
from langchain.tools import tool
import torch

@tool
def img_pipeline(query: str):
    """ Image Generating tool  """

    try:
        # Load Diffusion Image Model
        device = "cude" if torch.cude.is_available() else "cpu"
        img_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        ).to(device)        
        
        prompt = query

        image = img_pipe(prompt).images[0]

        return image
    except Exception as e:
        raise ValueError(f" :( Error in Generating Image: {e} ")


