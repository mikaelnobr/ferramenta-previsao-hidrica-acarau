import pandas as pd
import matplotlib.pyplot as plt
import os

print("--- INICIANDO GERAÇÃO DE GRÁFICOS DE ANÁLISE ---")

# 1. Definir os caminhos e criar a pasta para salvar os gráficos
PROCESSED_DATA_PATH = os.path.join('data', 'processed')
GRAPHS_PATH = 'graficos_analise/'
os.makedirs(GRAPHS_PATH, exist_ok=True)
print(f"Os gráficos serão salvos na pasta: '{GRAPHS_PATH}'")

CENARIOS = ['baixo', 'medio', 'alto']

# 2. Loop para gerar um gráfico para cada cenário
for cenario in CENARIOS:
    print(f"\nGerando gráfico para o cenário: {cenario.upper()}")
    
    arquivo_path = os.path.join(PROCESSED_DATA_PATH, f'{cenario}_consolidado_diario.csv')
    
    if not os.path.exists(arquivo_path):
        print(f"  AVISO: Arquivo '{arquivo_path}' não encontrado. Pulando este cenário.")
        continue
    
    # Carrega os dados processados, garantindo que a data seja o índice
    df = pd.read_csv(arquivo_path, index_col=0, parse_dates=True)
    
    # Encontra automaticamente todas as colunas de vazão
    vazao_cols = [col for col in df.columns if 'vazao' in col]
    
    if not vazao_cols:
        print(f"  AVISO: Nenhuma coluna de vazão encontrada no arquivo do cenário '{cenario}'.")
        continue

    # 3. Criação do Gráfico
    plt.style.use('dark_background') # Estilo escuro para combinar com o VS Code
    plt.figure(figsize=(15, 7)) # Define um bom tamanho para o gráfico
    
    for coluna in vazao_cols:
        # Pega um nome mais limpo para a legenda
        nome_estacao = coluna.replace('vazao_', '').replace('_', ' ').title()
        plt.plot(df.index, df[coluna], label=nome_estacao)
        
    # 4. Customização e Salvamento
    plt.title(f"Vazão Diária - Cenário de {cenario.replace('_', ' ').title()} Precipitação", fontsize=16)
    plt.xlabel("Data", fontsize=12)
    plt.ylabel("Vazão (m³/s)", fontsize=12)
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout() # Ajusta o gráfico para caber tudo
    
    # Salva o gráfico como uma imagem na pasta criada
    caminho_salvar = os.path.join(GRAPHS_PATH, f"vazao_diaria_{cenario}.png")
    plt.savefig(caminho_salvar)
    print(f"  Gráfico salvo em: '{caminho_salvar}'")
    
    # Mostra o gráfico na tela
    plt.show()

print("\n--- GERAÇÃO DE GRÁFICOS CONCLUÍDA ---")