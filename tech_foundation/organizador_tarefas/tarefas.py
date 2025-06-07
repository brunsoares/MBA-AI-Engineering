import json

# Função para adicionar uma tarefa
def salvar_tarefas():
    with open('tarefas.json', 'w') as arquivo:
        json.dump(tarefas, arquivo, indent=4)

# Função para carregar as tarefas do arquivo
def carregar_tarefas():
    try:
        with open('tarefas.json', 'r') as arquivo:
            return json.load(arquivo)
    except FileNotFoundError:
        print("Arquivo de tarefas não encontrado! Criando um novo arquivo.")
        return []

# Função para adicionar uma tarefa
def adicionar_tarefa():
    nome_tarefa = input("Digite o nome da tarefa: ")
    categoria_tarefa = input("Digite a categoria da tarefa: ")
    
    tarefa = {
        "nome": nome_tarefa,
        "categoria": categoria_tarefa,
        "concluida": False
    }
    tarefas.append(tarefa)
    print(f"Tarefa '{nome_tarefa}' adicionada com sucesso!")
    
    salvar_tarefas()

# Função para marcar uma tarefa como concluída
def marcar_tarefa_concluida():
    listar_tarefas()
    if not tarefas:
        print("Nenhuma tarefa para marcar como concluída.")
        return
    
    try:
        indice = int(input("Digite o número da tarefa a ser marcada como concluída: ")) - 1
        if 0 <= indice < len(tarefas):
            tarefas[indice]["concluida"] = True
            print(f"Tarefa '{tarefas[indice]['nome']}' marcada como concluída!")
            salvar_tarefas()
        else:
            print("Número da tarefa inválido.")
    except ValueError:
        print("Entrada inválida. Por favor, digite um número.")

# Função para listar as tarefas
def listar_tarefas():
    if not tarefas:
        print("Nenhuma tarefa encontrada.")
        return
    
    print("\nLista de Tarefas:")
    for i, tarefa in enumerate(tarefas, start=1):
        status = "Concluída" if tarefa["concluida"] else "Pendente"
        print(f"{i}. {tarefa['nome']} - Categoria: {tarefa['categoria']} - Status: {status}")


# Função para listar tarefas pendentes
def listar_tarefas_pendentes():
    pendentes = [tarefa for tarefa in tarefas if not tarefa["concluida"]]
    if not pendentes:
        print("Nenhuma tarefa pendente encontrada.")
        return
    
    print("\nTarefas Pendentes:")
    for i, tarefa in enumerate(pendentes, start=1):
        print(f"{i}. {tarefa['nome']} - Categoria: {tarefa['categoria']}")

# Função para listar por categoria
def listar_tarefas_por_categoria():
    categoria = input("Digite a categoria para filtrar as tarefas: ")
    filtradas = [tarefa for tarefa in tarefas if tarefa["categoria"].lower() == categoria.lower()]
    
    if not filtradas:
        print(f"Nenhuma tarefa encontrada na categoria '{categoria.capitalize()}'.")
        return
    
    print(f"\nTarefas na categoria '{categoria}':")
    for i, tarefa in enumerate(filtradas, start=1):
        status = "Concluída" if tarefa["concluida"] else "Pendente"
        print(f"{i}. {tarefa['nome']} - Status: {status}")


tarefas = []
carregar_tarefas()

while True:
    print("\nOrganizador de Tarefas")
    print("1. Adicionar Tarefa")
    print("2. Listar Tarefas")
    print("3. Marcar Tarefa como Concluída")
    print("4. Listar Tarefas Pendentes")
    print("5. Listar Tarefas por Categoria")
    print("6. Sair")
    escolha = input("Escolha uma opção: ")
    match escolha:
        case '1':
            adicionar_tarefa()
        case '2':
            listar_tarefas()
        case '3':
            marcar_tarefa_concluida()
        case '4':
            listar_tarefas_pendentes()
        case '5':
            listar_tarefas_por_categoria()
        case '6':
            print("Saindo do organizador de tarefas. Até logo!")
            break;
        case _:
            print("Opção inválida. Por favor, escolha uma opção válida.")
    salvar_tarefas()