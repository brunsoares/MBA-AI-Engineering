#!pip install diffusers transformers accelerate safetensors
from diffusers import StableDiffusionPipeline
import torch

# Verifica se existe uma GPU disponível
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Mover o modelo para dispositivo
pipe = pipe.to(device)

# Definição do prompt para geração
prompt = "Capybara in space wearing an astronaut suit, photorealistic" 

# Gerar a imagem com o prompt
image = pipe(prompt).images[0]

# Visualização da imagem
image