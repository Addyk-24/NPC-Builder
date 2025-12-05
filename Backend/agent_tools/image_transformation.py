from transformers import pipeline
import numpy as np
from uuid import uuid4 
from PIL import Image

depth_model_path = "LiheYoung/depth-anything-small-hf"


depth_estimator = pipeline(
    task="depth-estimation",
    model= depth_model_path
)

def generate_depth_scale(image_path):

    image = Image.open(image_path)

    # Generating depth scale

    depth = depth_estimator(image)


    depth_map = depth["depth"] # PIL Imagess

    depth_array = np.array(depth["predicted_depth"]) # Converted numpy array

    # Normalize depth to 0-1 range
    depth_normalized = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min())

    # Save depth visualization
    depth_img = Image.fromarray((depth_normalized * 255).astype(np.uint8))
    depth_filename = f"depth_{uuid4()}.png"
    depth_img.save(depth_filename)



    return depth_normalized

def normal_map_from_depth(depth_array: np.ndarray, depth_scale: float = 1.0) -> np.ndarray:
    """
    Generates a normal map from a given depth map

    Args:
        depth_array (np.ndarray): A 2D numpy array representing the depth map.
    Returns:
        np.ndarray: A 3D numpy array representing the normal map.
    """

    # Apply depth scaling
    depth_scaled = depth_array * depth_scale

    dz_dx, dz_dy  = np.gradient(depth_array)

# The '1' represents the Z-component, assuming depth increases along Z

    normals = np.dstack([-dz_dx,-dz_dy, np.ones_like(depth_array)])

    norm = np.linalg.norm(normals, axis=2, keepdims=True)

    normals_normalized = normals / (norm + 1e-8)

    normal_map_rgb = ((normals_normalized + 1.0) / 127.5).astype(np.uint8)

    return normal_map_rgb



    

