# Install the necessary libraries
# !pip install --upgrade diffusers transformers -q
# !pip install torch -q

import torch

# Check if CUDA is available and print the result
if torch.cuda.is_available():
    print("CUDA is available. PyTorch will use the GPU.")
else:
    print("CUDA is not available. PyTorch will use the CPU.")

# Sample code to verify GPU usage
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Example tensor operation to verify CUDA usage
x = torch.tensor([1.0, 2.0, 3.0], device=device)
print(f"Tensor on {device}: {x}")


from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib as plt
import cv2

class CFG:
    import torch
    print(torch.version.cuda)
    device = "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps=35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size=(400,400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_set = "6"
    prompt_max_length=12


from diffusers import StableDiffusionPipeline

# Load the model with CPU support
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    revision="fp16",
    use_auth_token="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",
    guidance_scale=9
)

# Set the device to CPU
image_gen_model = image_gen_model.to("cpu")

# If CFG.device is set to "cpu", make sure to update the code accordingly
CFG.device = "cpu"

def generate_image(prompt,model):
  image = model(prompt,num_inference_steps=CFG.image_gen_steps,generator=CFG.generator).images[0]
  image = image.resize(CFG.image_gen_size)
  return image

generate_image("two trains crossing each other",image_gen_model)

generate_image("lion with sunglasses",image_gen_model)
