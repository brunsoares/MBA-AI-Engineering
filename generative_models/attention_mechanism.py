import numpy as np
import math
import matplotlib.pyplot as plt
np.set_printoptions(precision=3,suppress=True)

# Função para Embedding de token e D_MODEL
def embedding_token_dmodel(tokens, d_model, seed=42):
  rng = np.random.default_rng(seed)
  table = {}
  for i in tokens:
    table[i] = rng.uniform(-1, 1, d_model).astype(np.float32)
  return table

# Execução de exemplos - embedding de token
tokens = ['carla', 'canta', 'canções', 'calmas', 'no', 'campo']
d_model = 6
embedding_table = embedding_token_dmodel(tokens, d_model)

# Função para embedding por posição
def positional_embedding(seq_len, d_model):
    if d_model % 2 != 0:
        raise ValueError("D_MODEL deve ser para para encoding")
    pe = np.zeros((seq_len, d_model), dtype=np.float32)

    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            # Decay = 10000
            denom = 10000 ** (i / d_model)
            pe[pos, i] = math.sin(pos / denom)
            pe[pos, i + 1] = math.cos(pos / denom)
    return pe

# Execução dos exemplos
d_model = 6
seq_len = len(tokens)
pe = positional_embedding(seq_len, d_model)

# Composição final do embedding por transformer - Token + posição
emb_transformer = np.stack([embedding_table[t] for t in tokens], axis=0)
emb_in = emb_transformer + pe
print(f'Embeddings finais: {emb_in.shape}')
print(f'Primeiro token: {emb_in[0]}')

# Execução do mecanismo de atenção
# Função softmax
def softmax(x):
  x = x - x.max(axis=-1, keepdims=True)
  e = np.exp(x)
  return e / e.sum(axis=-1, keepdims=True)

# Função para Produto escalar escalonado
def scaled_dot_product_attentition(e):
  query = e; keys = e; value = e; # Criação das cópias
  d_model = e.shape[1] # Definição do d_model
  scores = query @ keys.T # Multiplicação de query pela matriz transposta keys
  scaled = scores / math.sqrt(d_model) # Normalização dos dados
  weights = softmax(scaled) # Aplicação da função softmax
  out = weights @ value # Multiplicação final
  return scores, scaled, weights, out

# Execução do mecanismo
scores, scaled, weights, out = scaled_dot_product_attentition(emb_in)
print(f'Scores: {scores.shape}')
print(f'Scaled: {scaled.shape}')
print(f'Weights: {weights.shape}')
print(f'Out: {out.shape}')

# Visualização
plt.figure(figsize=(5, 4))
plt.imshow(weights, cmap='viridis')
plt.title('Pesos de atenção')
plt.xlabel('Keys')
plt.ylabel('Query')
plt.colorbar()
plt.tight_layout()
plt.show()