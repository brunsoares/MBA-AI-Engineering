import numpy as np
np.set_printoptions(precision=3,suppress=True)

# Função para Embedding de token e D_MODEL
def embedding_token_dmodel(tokens, d_model, seed=42):
  rng = np.random.default_rng(seed)
  table = {}
  for i in tokens:
    table[i] = rng.uniform(-1, 1, d_model).astype(np.float32)
  return table

# Execução de exemplos
tokens = ['carla', 'canta', 'canções', 'calmas', 'no', 'campo']
d_model = 6
embedding_table = embedding_token_dmodel(tokens, d_model)
word = 'canta'
print(f'Gerados os embeddings para {len(tokens)} tokens (D_MODEL={d_model})')
print(f'Embedding de {word}: {embedding_table[word]}')
print(f'Dimensão: {embedding_table[word].shape}')

context_win_size = 5
print(f'Janela de contexto: {context_win_size} tokens -> {tokens[:context_win_size]}')
