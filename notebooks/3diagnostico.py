import pandas as pd
import os

# --- CONFIGURE AQUI ---
# Edite as duas linhas abaixo para apontar para um dos seus arquivos de dados brutos.
# Escolha um que você sabe que tem 1 ano de dados.

cenario = 'alto' 
nome_do_arquivo = '35235000-VARZEA DO GROSSO.csv' 

# --------------------------------------------------------------------------------

# O resto do script é automático
arquivo_para_verificar = os.path.join('data', 'raw', f'cenario_{cenario}', nome_do_arquivo)

print(f"--- DIAGNÓSTICO DE VALORES: {arquivo_para_verificar} ---")

try:
    # Tenta ler o arquivo com as configurações que já descobrimos
    df = pd.read_csv(arquivo_para_verificar, encoding='latin-1', sep=';')
    if 'Data' not in df.columns or 'Hora' not in df.columns:
        df = pd.read_csv(arquivo_para_verificar, encoding='latin-1', sep=',')
    
    print(f"\nArquivo lido com sucesso. Total de linhas: {len(df)}")

    # Lista das colunas que deveriam conter apenas números
    colunas_para_checar = ['Chuva (mm)', 'Nível (cm)', 'Vazão (m3/s)']

    for coluna in colunas_para_checar:
        if coluna not in df.columns:
            print(f"\nAVISO: A coluna '{coluna}' não foi encontrada no arquivo.")
            continue

        print(f"\n--- Verificando a coluna: '{coluna}' ---")
        
        # Converte a coluna para tipo numérico. 'errors='coerce'' transforma 
        # qualquer valor que não seja um número em um valor inválido (NaN).
        # Adicionado .astype(str).str.replace(',', '.') para lidar com vírgulas decimais.
        numeric_series = pd.to_numeric(df[coluna].astype(str).str.replace(',', '.'), errors='coerce')
        
        # Encontra as linhas onde a conversão para número falhou.
        # A condição é: o valor original não estava vazio, mas depois da conversão ele se tornou inválido.
        falhas = df[numeric_series.isnull() & df[coluna].notnull()]
        
        num_falhas = len(falhas)
        print(f"  - Número de valores que NÃO são números válidos: {num_falhas}")

        if num_falhas > 0:
            # Mostra quais são os valores únicos que estão causando o problema
            valores_problematicos = falhas[coluna].unique()
            print(f"  - AMOSTRA DE VALORES PROBLEMÁTICOS ENCONTRADOS:")
            # Mostra os 10 primeiros valores únicos que deram erro
            for valor in valores_problematicos[:10]: 
                print(f"    -> '{valor}'")

except FileNotFoundError:
    print(f"\nERRO: O arquivo '{arquivo_para_verificar}' não foi encontrado. Verifique as variáveis 'cenario' e 'nome_do_arquivo' no topo do script.")
except Exception as e:
    print(f"\nOcorreu um erro inesperado durante a execução: {e}")