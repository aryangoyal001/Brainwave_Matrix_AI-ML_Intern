# Install necessary libraries (uncomment if running in Colab or fresh environment)
# !pip install diffusers transformers torch --upgrade -q

import torch
from diffusers import StableDiffusionPipeline

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device.upper()}")

# Configuration
class CFG:
    seed = 42
    generator = torch.Generator(device=device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    device = device

# Load model with correct revision (fp16 only works with GPU)
image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id,
    torch_dtype=torch.float32,  # Use float32 on CPU
    revision="main"             # Use "main" for CPU (not "fp16")
)

# Move to CPU or GPU
image_gen_model = image_gen_model.to(CFG.device)

# Image generation function
def generate_image(prompt, model):
    print(f"Generating image for prompt: {prompt}")
    image = model(prompt, 
                  num_inference_steps=CFG.image_gen_steps,
                  guidance_scale=CFG.image_gen_guidance_scale,
                  generator=CFG.generator).images[0]
    image = image.resize(CFG.image_gen_size)
    return image

# Example prompts
img1 = generate_image("two trains crossing each other", image_gen_model)
img2 = generate_image("lion with sunglasses", image_gen_model)

# Save or show images
img1.show()
img2.show()
# img1.save("train_crossing.png")
# img2.save("lion_sunglasses.png")
