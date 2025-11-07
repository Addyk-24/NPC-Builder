from diffusers import DiffusionPipeline
from langchain.tools import tool


@tool
def img_pipeline(query: str):
    """  """
    img_pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
    prompt = query

    image = img_pipe(prompt).images[0]

    return image


