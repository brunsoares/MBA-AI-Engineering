import numpy as np
import math
import matplotlib.pyplot as plt
np.set_printoptions(precision=3,suppress=True)

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
tokens = ['carla', 'canta', 'canções', 'calmas', 'no', 'campo']
d_model = 6
seq_len = len(tokens)
pe = positional_embedding(seq_len, d_model)
print(f'Shape PE: {pe.shape}')
print(f'Primeiras posições e dimensões (3x6), linhas invertidas: \n {pe[:3,:6][::-1]}')

# Visualização gráfica
plt.figure(figsize=(6, 3))
plt.imshow(pe.T[::-1], aspect='auto', origin='lower')
plt.title('Embedding por posição (seno/cosseno)')
plt.xlabel('POSIÇÃO')
plt.ylabel('DIMENSÃO')
plt.colorbar()
plt.tight_layout()
plt.show()