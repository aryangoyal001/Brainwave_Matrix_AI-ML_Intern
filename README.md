# 🖼️ Text-to-Image Generator using Stable Diffusion

This is my **Text-to-Image Generation Task** for the **AI/Machine Learning** Internship.  
This Python app takes a **text prompt** (e.g., “a cat playing piano on Mars”) and generates a stunning **AI-generated image** using the **Stable Diffusion model**.

---

## 📌 Project Description

I built a **Text-to-Image Generation App** using **Python** and **Hugging Face’s Diffusers library**.  
This project demonstrates the capabilities of **Generative AI** by turning any custom text into a visually rich image.  
The Stable Diffusion model used here runs even on **CPU**, making it accessible without high-end GPUs.

---

## 🎯 Goals

- Utilize **Stable Diffusion** for creative image generation.
- Build a working app that takes **text input** and returns an image.
- Ensure compatibility with CPU (for easier local use).
- Learn to work with **Hugging Face Diffusers**, **Torch**, and **transformers**.

---

## 🌟 Key Features

- 🔤 **Prompt-Based Image Generation** – Just type what you want to see.
- ⚙️ **Runs on CPU** – No GPU required.
- 📐 **Custom Image Resizing** – Set your preferred output size.
- 🔁 **Consistent Results** – Seed-based generation using `torch.Generator`.
- 📷 **Multiple Test Prompts** – Try anything from realistic to fantasy!

---

## 🧪 Technologies Used

- **Python**
- **Hugging Face Diffusers** – for Stable Diffusion model
- **Torch** – for model operations and tensor management
- **Transformers** – for prompt embedding
- **PIL** – for image resizing and display

---

## 🔁 How It Works

1. Import the necessary libraries and check CUDA availability.
2. Load the **Stable Diffusion model** using `StableDiffusionPipeline`.
3. Configure seed and device (CPU in this case).
4. Generate an image using a user prompt and predefined steps.
5. Display or save the output image.

---

## 📁 Project Structure
```
text-to-image-generator/
│
├── text_to_image.py   # Main script for image generation
├── README.md          # You're reading it!
└── generated_images/  # (Optional) Folder to store generated outputs
```
---

## 🧠 Sample Prompts & Outputs

- `"two trains crossing each other"` → 🚂🚂
- `"lion with sunglasses"` → 🦁🕶️
- `"a panda DJing in space"` → 🐼🎧🌌

---

## 📌 Author

**Aryan Goyal**  
_AI / Machine Learning / Python Developer_

🔗 [LinkedIn Profile](https://www.linkedin.com/in/aryan-goyal-96077a298/)  
📧 [aryangoyal2352007@gmail.com](mailto:aryangoyal2352007@gmail.com)

---

## 🏷️ Tags

`#Python` `#MachineLearning` `#GenerativeAI` `#StableDiffusion` `#Diffusers` `#ImageGeneration` `#AI` `#AryanGoyalProjects`

---
