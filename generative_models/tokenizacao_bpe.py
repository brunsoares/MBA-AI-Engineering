import numpy as np
import collections
import math
import matplotlib.pyplot as plt
np.set_printoptions(precision=3,suppress=True)

# Função para tokenização simplificada do BPE
def token_bpe_simplified(text, num_merge=5):
  print(f'--- Tokenização BPE simplificada com {num_merge} merges ---')
  # Primeiro deixamos em minúsculo
  text_lower = text.lower()
  print(f'(1) Texto em minúsculo: {text_lower}')
  # Separamos em caracteres iniciais
  tokens = list(text_lower)
  print(f'(2) Caracteres iniciais: {tokens}')

  merge_history = []
  for step in range(num_merge):
    # Contamos os pares adjacentes
    pair_counts = collections.Counter()
    for i in range(len(tokens) - 1):
      pair = (tokens[i], tokens[i + 1])
      pair_counts[pair] += 1

    if not pair_counts:
      print(f'Sem pares para mesclar, encerrando tokenização')
      break

    # Encontramos o par mais frequente
    (pair_to_merge, freq) = pair_counts.most_common(1)[0]
    print(f'--- Merge {step+1}: par mais frequente: {pair_to_merge} (freq={freq})---')
    merge_history.append(pair_to_merge)

    # Mesclamos os pares - Aplicando fusão
    new_tokens = []
    i = 0
    while i < len(tokens):
      if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair_to_merge:
        new_tokens.append(pair_to_merge[0] + pair_to_merge[1])
        i += 2
      else:
        new_tokens.append(tokens[i])
        i += 1
    
    tokens = new_tokens
    print(f'(3) Tokens após merge: {tokens}')

  print(f'--- Tokenização finalizada ---')
  print(f'Merge history: {merge_history}')
  return tokens, merge_history

# Execução dos exemplos
# 000 - Vários "ca" e "cam" -> Merges maiores mostram agrupamento por sílabas repetidas
text_input_000 = "Carla canta canções calmas no campo"

tokens_000, merge_history_000 = token_bpe_simplified(text_input_000, num_merge=5)
print(f'Tokens: {tokens_000}')

# 001 - Poucos caracteres e sílabas repetidas - merges pequenos formam palavras completas
text_input_001 = "João viu sol e mar"

tokens_001, merge_history_001 = token_bpe_simplified(text_input_001, num_merge=3)
print(f'Tokens: {tokens_001}')