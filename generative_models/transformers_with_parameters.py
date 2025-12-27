#!pip install -q transformers accelerate torch --extra-index-url https://download.pytorch.org/whl/cu121

from transformers import pipeline, set_seed
import torch, textwrap, pprint

# Definição do dispositivo e funções auxiliares
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Dispositivo:", device)

# Função para printar o conteúdo
def print_content(title, content):
    print('\n' + "="*80)
    print(title)
    print("="*80)
    print(textwrap.fill(content, width=110))


# Função para mostrar os parâmetros
def show_params(params):
    print("\nParâmetros:")
    pprint.pprint(params)


# Criando pipeline de geração
model_id="pierreguillou/gpt2-small-portuguese" # Modelo opcional pequeno
# model_id="gpt2" # Modelo padrão gpt2
# model_id="wandgibaut/periquito-3B"
# model_id="artificialguybr/OpenHermesV2-PTBR"

generator = pipeline(
    "text-generation",
    model=model_id,
    device=0 if device=="cuda" else -1,
)

print("Pipeline criado com:", model_id)

# Função para geração dinâmica
def generate_text(
    prompt,
    max_new_tokens=60,
    do_sample=True,
    temperature=None,
    top_k=None,
    top_p=None,
    num_beams=None,
    seed=None
):
  params = {
      "max_new_tokens": max_new_tokens,
      "do_sample": do_sample
  }

  if temperature is not None:
    params["temperature"] = temperature

  if top_k is not None:
    params["top_k"] = top_k
  
  if top_p is not None:
    params["top_p"] = top_p

  if num_beams is not None: # Geralmente em beam search não amostramos
    params["num_beams"] = num_beams
    params["do_sample"] = False

  if seed is not None:
    set_seed(seed)

  show_params(params)
  output = generator(prompt, **params)
  text = output[0]["generated_text"]
  print_content("PROMPT", prompt)
  print_content("OUTPUT", text)
  
  return text

# Testando em diversos cenários
# EX: 0 - Sanity Check
ex000 = generate_text(
    prompt="A inteligência artificial no Brasil está avançando porque",
    max_new_tokens=50,
    do_sample=True,
    temperature=0.7,
    seed=42
)

# Ex: 1 - Criatividade x Precisão
ex001 = generate_text(
    prompt="O futuro da inteligência artificial no Brasil será marcado por",
    temperature=0.2,
    seed=7
)

ex002 = generate_text(
    prompt="O futuro da inteligência artificial no Brasil será marcado por",
    temperature=1.0,
    seed=7
)

ex0003 = generate_text(
    prompt="O futuro da inteligência artificial no Brasil será marcado por",
    temperature=1.5,
    seed=7
)

# EX: 2 - Limite de palavras candidatas (TOP_K)
ex003 = generate_text(
    prompt="Em 2050, as cidades brasileiras serão",
    temperature=0.8,
    top_k=5,
    seed=21
)

ex004 = generate_text(
    prompt="Em 2050, as cidades brasileiras serão",
    temperature=0.8,
    top_k=50,
    seed=21
)

ex005 = generate_text(
    prompt="Em 2050, as cidades brasileiras serão",
    temperature=0.8,
    top_k=100,
    seed=21
)

# EX: 3 - Controle adaptativo de diversidade (TOP_P)
ex006 = generate_text(
    prompt="A educação do futuro será baseada em",
    temperature=0.8,
    top_p=0.8,
    seed=10
)

ex007 = generate_text(
    prompt="A educação do futuro será baseada em",
    temperature=0.8,
    top_p=0.95,
    seed=10
)

ex008 = generate_text(
    prompt="A educação do futuro será baseada em",
    temperature=0.8,
    top_p=0.99,
    seed=10
)

# EX: 4 - Deterministico x Amostragem (DO_SAMPLE)
ex009 = generate_text(
    prompt="O maior desafio da humanidade no século XXI é",
    do_sample=False
)

ex010 = generate_text(
    prompt="O maior desafio da humanidade no século XXI é",
    do_sample=True,
    temperature=0.7,
    seed=42
)

ex011 = generate_text(
    prompt="O maior desafio da humanidade no século XXI é",
    do_sample=True,
    temperature=0.7,
    seed=90
)

#EX: 5 - Beam Search
ex012 = generate_text(
    prompt="O Brasil em 2100 será um país",
    num_beams=3,
    max_new_tokens=80,
)

ex013 = generate_text(
    prompt="O Brasil em 2100 será um país",
    num_beams=5,
    max_new_tokens=80
)

#Ex: 6 - Combinação de parâmetros
ex014 = generate_text(
    prompt="No ano de 3000, a vida na Terra será",
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
    seed=3000
)

ex015 = generate_text(
    prompt="No ano de 3000, a vida na Terra será",
    do_sample=True,
    temperature=0.7,
    top_k=50,
    seed=3000
)

ex016 = generate_text(
    prompt="No ano de 3000, a vida na Terra será",
    do_sample=False,
    num_beams=5
)