import pandas as pd
import os
import json

print("--- FASE 1: INICIANDO PROCESSAMENTO DE DADOS REAIS ---")

try:
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    estacoes_info = config['estacoes']
except FileNotFoundError:
    print("ERRO CRÍTICO: O arquivo 'config.json' não foi encontrado.")
    exit()

CENARIOS = ['baixo', 'medio', 'alto']
RAW_DATA_PATH = os.path.join('data', 'raw')
PROCESSED_DATA_PATH = os.path.join('data', 'processed')
os.makedirs(PROCESSED_DATA_PATH, exist_ok=True)

for cenario in CENARIOS:
    print(f"\n--- Processando Cenário: {cenario.upper()} ---")
    lista_dfs_estacoes = []
    
    for nome_arquivo, info in estacoes_info.items():
        arquivo_path = os.path.join(RAW_DATA_PATH, f'cenario_{cenario}', nome_arquivo)
        if not os.path.exists(arquivo_path):
            continue
            
        print(f"  Lendo arquivo: {arquivo_path}")
        try:
            df_estacao = pd.read_csv(arquivo_path, encoding='latin-1', sep=';')
            if 'Data' not in df_estacao.columns or 'Hora' not in df_estacao.columns:
                 df_estacao = pd.read_csv(arquivo_path, encoding='latin-1', sep=',')
        except Exception as e:
            print(f"  ERRO ao ler o arquivo {nome_arquivo}: {e}. Pulando.")
            continue

        id_estacao = nome_arquivo.replace('.csv', '').lower().replace(' ', '_').replace('-', '_')

        colunas_reais = {'Chuva (mm)': f'chuva_{id_estacao}', 'Nível (cm)': f'nivel_{id_estacao}', 'Vazão (m3/s)': f'vazao_{id_estacao}'}
        for col_original, col_nova in colunas_reais.items():
            if col_original in df_estacao.columns:
                df_estacao[col_nova] = df_estacao[col_original].astype(str).str.strip()
        
        df_estacao['timestamp'] = pd.to_datetime(df_estacao['Data'] + ' ' + df_estacao['Hora'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
        df_estacao.dropna(subset=['timestamp'], inplace=True)
        df_estacao = df_estacao.set_index('timestamp')
        
        # Seleciona apenas as colunas que foram renomeadas
        colunas_renomeadas = [col for col in colunas_reais.values() if col in df_estacao.columns]
        lista_dfs_estacoes.append(df_estacao[colunas_renomeadas])
        
    if not lista_dfs_estacoes:
        print(f"  Nenhum dado encontrado para o cenário '{cenario}'. Pulando.")
        continue
        
    # Concatena todos os dataframes de estações. O Pandas alinha pelo timestamp.
    df_consolidado = pd.concat(lista_dfs_estacoes, axis=1)
    
    print(f"  Dados consolidados. Período: de {df_consolidado.index.min()} a {df_consolidado.index.max()}")

    print("  Tratando valores numéricos e dados ausentes...")
    agg_funcs = {}
    for nome_arquivo in estacoes_info.keys():
        id_estacao = nome_arquivo.replace('.csv', '').lower().replace(' ', '_').replace('-', '_')
        
        for tipo_dado in ['chuva', 'nivel', 'vazao']:
            coluna = f'{tipo_dado}_{id_estacao}'
            if coluna in df_consolidado.columns:
                # Converte para número, limpando espaços e trocando vírgula por ponto
                df_consolidado[coluna] = pd.to_numeric(df_consolidado[coluna].astype(str).str.replace(',', '.'), errors='coerce')
                
                # Preenche os vazios (interpolate para contínuos, fillna(0) para chuva)
                if tipo_dado != 'chuva':
                    df_consolidado[coluna] = df_consolidado[coluna].interpolate(method='linear')
                else:
                    df_consolidado[coluna] = df_consolidado[coluna].fillna(0)
                
                agg_funcs[coluna] = 'sum' if tipo_dado == 'chuva' else 'mean'
            
    if not agg_funcs:
        print(f"  ERRO: Nenhuma coluna para agregar foi encontrada.")
        continue

    print("  Agregando dados para a média diária...")
    # Resample cria um índice diário completo, mesmo para dias sem dados, que serão NaN
    df_diario = df_consolidado.resample('D').agg(agg_funcs)
    
    # Re-preenchemos possíveis lacunas criadas pelo resample
    for col in df_diario.columns:
        if 'chuva' not in col:
            df_diario[col] = df_diario[col].interpolate(method='linear')
    df_diario.fillna(0, inplace=True) # Preenche qualquer lacuna restante com 0

    for nome_arquivo, info in estacoes_info.items():
        id_estacao = nome_arquivo.replace('.csv', '').lower().replace(' ', '_').replace('-', '_')
        df_diario[f'distancia_{id_estacao}'] = info['distancia_km']
    
    output_path = os.path.join(PROCESSED_DATA_PATH, f'{cenario}_consolidado_diario.csv')
    df_diario.to_csv(output_path)
    print(f"  --> Cenário {cenario} salvo com sucesso! Total de dias processados: {len(df_diario)}")

print("\n--- FASE 1 CONCLUÍDA! ---")