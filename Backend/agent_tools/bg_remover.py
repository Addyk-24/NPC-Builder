import rembg
import numpy as np
from PIL import Image
from langchain.tools import tool

@tool
def background_remover(image : Image.Image):

    try:
        # Loading Image
        input_img = Image.open(image)

        # Convertin Image to Array
        img_array = np.array(input_img)

        # BG Remove
        output_array = rembg.remove(img_array)

        # New Image without BG
        output_image = Image.fromarray(output_array)

        # Save the output image
        output_image.save('output_image.jpg')

        return output_image
    
    except Exception as e:
        raise ValueError(f" :( Background Remover temporarily unavailable. Error: {e} ") 

