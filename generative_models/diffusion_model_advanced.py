#!pip install -q diffusers transformers accelerate safetensors torch --extra-index-url https://download.pytorch.org/whl/cu121
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import time

# Verifica dispositivo
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device)

# Funcionamento do pipeline
## Text encoder (CLIP) - transforma o prompt em embeddings
## U-Net faz um denoising iterativo no espaço latente (latent diffusion)
## VAE Decoder reconstrói a imagem final a partir do latente
## Scheduler define a política de remoção de ruído a cada passo

# Carregando o modelo
## torch_dtype=float16 reduz memória/tempo - exige gpu com suporte a FP16
## enable_attention_slicing reduz pico de memória
model_id = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

# Substitui o scheduler padrão por um mais rápido/estável
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(device)

if device == "cuda":
    pipe.enable_attention_slicing()

print("Modelo carregado")

# Parâmetros
## negative_prompt - O que evitar na imagem
## num_inference_steps - mais passos - mais qualidade e tempo
## guidance_scale - quão fortemente vai seguir o texto do prompt
## seed - para repodutibilidade, mudando para variar resultados
prompt = "Capybara in space wearing an astronaut suit, extremely detailed and with realistic lighting"
negative_prompt = "Low quality, blurry, body deformations, overexposed text"

num_inference_steps = 30
guidance_scale = 9.0
seed = 12341

# Geração da imagem
generator = torch.Generator(device=device).manual_seed(seed)
start = time.time()
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=num_inference_steps,
    guidance_scale=guidance_scale,
    generator=generator,
).images[0]

elapsed = time.time() - start
print(f"Tempo de geração: {elapsed:.1f}s")

display(image)