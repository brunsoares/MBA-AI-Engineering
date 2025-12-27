#!pip install transformers

from transformers import pipeline

# Criando modelo
generator = pipeline('text-generation', model='gpt2')

# Criação do prompt
prompt = "The winner of the 2014 World Cup was the national team of "

# Geração do output
output = generator(prompt, max_length=50, num_return_sequences=1)
print(output[0]["generated_text"])