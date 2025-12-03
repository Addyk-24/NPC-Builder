from transformers import pipeline
import numpy as np
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

    # depth["depth"] is PIL Image (grayscale)
    # depth["predicted_depth"] is numpy array with raw values

    depth_map = depth["depth"]

    depth_array = np.array(depth["predicted_depth"])

    return depth_map, depth_array
