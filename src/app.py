import pandas as pd
import numpy as np
import json
import joblib
import os
import streamlit as st
from tensorflow.keras.models import load_model # type: ignore
import plotly.graph_objects as go
from datetime import timedelta

# --- CONFIGURAÇÃO INICIAL E CARREGAMENTO DE ARTEFATOS ---
st.set_page_config(page_title="Gestão Hídrica Acaraú", layout="wide")

@st.cache_resource
def carregar_tudo():
    """Carrega modelos, scalers e configurações de uma vez."""
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
    except FileNotFoundError:
        st.error("ERRO CRÍTICO: Arquivo 'config.json' não foi encontrado na raiz do projeto.")
        return None, None, None
    
    modelos, scalers = {}, {}
    for cenario in ['baixo', 'medio', 'alto']:
        if os.path.exists(f'models/modelo_{cenario}.h5') and os.path.exists(f'models/scaler_{cenario}.pkl'):
            modelos[cenario] = load_model(f'models/modelo_{cenario}.h5')
            scalers[cenario] = joblib.load(f'models/scaler_{cenario}.pkl')
            
    return config, modelos, scalers

config, modelos, scalers = carregar_tudo()
if config:
    estacoes_info = config.get('estacoes', {})
N_PAST = 30 

# --- FUNÇÕES DE BACKEND ---

def processar_arquivos_brutos(arquivos_carregados, estacoes_info):
    """Processa os arquivos CSV brutos carregados pelo usuário em memória."""
    with st.spinner("Processando e limpando os dados brutos..."):
        lista_dfs_estacoes = []
        mapa_arquivos_carregados = {f.name: f for f in arquivos_carregados}

        for nome_arquivo_esperado, info in estacoes_info.items():
            if nome_arquivo_esperado not in mapa_arquivos_carregados:
                st.error(f"Arquivo necessário não encontrado: '{nome_arquivo_esperado}'.")
                return None
            
            arquivo = mapa_arquivos_carregados[nome_arquivo_esperado]
            st.write(f"  Lendo arquivo: {arquivo.name}")
            try:
                df_estacao = pd.read_csv(arquivo, encoding='latin-1', sep=';')
                if 'Data' not in df_estacao.columns or 'Hora' not in df_estacao.columns:
                    df_estacao = pd.read_csv(arquivo, encoding='latin-1', sep=',')
            except Exception as e:
                st.error(f"Erro ao ler o arquivo {arquivo.name}: {e}")
                return None

            id_estacao = nome_arquivo_esperado.replace('.csv', '').lower().replace(' ', '_').replace('-', '_')

            colunas_reais = {'Chuva (mm)': f'chuva_{id_estacao}', 'Nível (cm)': f'nivel_{id_estacao}', 'Vazão (m3/s)': f'vazao_{id_estacao}'}
            for col_original, col_nova in colunas_reais.items():
                if col_original in df_estacao.columns:
                    df_estacao[col_nova] = df_estacao[col_original].astype(str).str.strip()
            
            df_estacao['timestamp'] = pd.to_datetime(df_estacao['Data'] + ' ' + df_estacao['Hora'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            df_estacao.dropna(subset=['timestamp'], inplace=True)
            df_estacao = df_estacao.set_index('timestamp')
            
            colunas_renomeadas = [col for col in colunas_reais.values() if col in df_estacao.columns]
            lista_dfs_estacoes.append(df_estacao[colunas_renomeadas])
            
        df_consolidado = pd.concat(lista_dfs_estacoes, axis=1)
        
        agg_funcs = {}
        for nome_arquivo in estacoes_info.keys():
            id_estacao = nome_arquivo.replace('.csv', '').lower().replace(' ', '_').replace('-', '_')
            for tipo_dado in ['chuva', 'nivel', 'vazao']:
                coluna = f'{tipo_dado}_{id_estacao}'
                if coluna in df_consolidado.columns:
                    df_consolidado[coluna] = pd.to_numeric(df_consolidado[coluna].astype(str).str.replace(',', '.'), errors='coerce')
                    if tipo_dado != 'chuva':
                        df_consolidado[coluna] = df_consolidado[coluna].interpolate(method='linear')
                    else:
                        df_consolidado[coluna] = df_consolidado[coluna].fillna(0)
                    agg_funcs[coluna] = 'sum' if tipo_dado == 'chuva' else 'mean'
                
        df_diario = df_consolidado.resample('D').agg(agg_funcs)
        
        for col in df_diario.columns:
            if 'chuva' not in col:
                df_diario[col] = df_diario[col].interpolate(method='linear')
        df_diario.fillna(0, inplace=True)

        for nome_arquivo, info in estacoes_info.items():
            id_estacao = nome_arquivo.replace('.csv', '').lower().replace(' ', '_').replace('-', '_')
            df_diario[f'distancia_{id_estacao}'] = info['distancia_km']
        
        st.success(f"Dados brutos processados com sucesso! Total de dias: {len(df_diario)}")
        return df_diario

def fazer_previsao_futura(df_input, modelo, scaler, dias_para_prever):
    ordem_colunas = scaler.feature_names_in_
    df_input = df_input.reindex(columns=ordem_colunas)
    dados_normalizados = scaler.transform(df_input)
    
    input_atual = dados_normalizados[-N_PAST:].reshape(1, N_PAST, dados_normalizados.shape[1])
    previsoes_normalizadas = []
    
    progresso = st.progress(0, text="Calculando previsões...")
    for i in range(dias_para_prever):
        previsao = modelo.predict(input_atual, verbose=0)
        previsoes_normalizadas.append(previsao[0])
        
        novo_registro = input_atual[0, -1, :].copy()
        target_indices = [list(ordem_colunas).index(col) for col in ordem_colunas if 'vazao' in col]
        for j, idx in enumerate(target_indices):
            novo_registro[idx] = previsao[0][j]
            
        input_atual = np.append(input_atual[:, 1:, :], [[novo_registro]], axis=1)
        progresso.progress((i + 1) / dias_para_prever, text=f"Calculando dia {i+1}/{dias_para_prever}")

    progresso.empty()
    dummy_array = np.zeros((len(previsoes_normalizadas), len(ordem_colunas)))
    for i, idx in enumerate(target_indices):
        dummy_array[:, idx] = [p[i] for p in previsoes_normalizadas]

    previsoes_desnormalizadas = scaler.inverse_transform(dummy_array)
    return previsoes_desnormalizadas[:, target_indices]

def interpolar_vazao(data_alvo, previsoes_df, estacoes_info):
    distancias = np.array([info['distancia_km'] for info in estacoes_info.values()])
    vazoes = previsoes_df.loc[data_alvo].values
    
    idx_sorted = np.argsort(distancias)
    distancias_sorted = distancias[idx_sorted]
    vazoes_sorted = vazoes[idx_sorted]
    
    return lambda dist_km: np.interp(dist_km, distancias_sorted, vazoes_sorted)

@st.cache_data
def convert_df_to_csv(df):
    """Função para converter o DataFrame para CSV, formatado para Excel em português."""
    return df.to_csv(index=True, index_label='Data', sep=';', decimal=',').encode('utf-8-sig')

# --- INTERFACE DO USUÁRIO ---
st.title("💧 Ferramenta Preditiva para Gestão Hídrica - Vale do Acaraú")

if not modelos:
    st.error("Nenhum modelo treinado foi encontrado na pasta 'models/'. Execute '2treinamento.py' primeiro.")
else:
    st.sidebar.header("Etapa 1: Previsão Geral")
    mapa_cenarios_display = {'Baixa Precipitação': 'baixo', 'Média Precipitação': 'medio', 'Alta Precipitação': 'alto'}
    cenarios_disponiveis = [key for key, val in mapa_cenarios_display.items() if val in modelos]
    cenario_display = st.sidebar.selectbox("1. Selecione o cenário climático:", cenarios_disponiveis)
    cenario_cod = mapa_cenarios_display[cenario_display]

    dados_brutos_files = st.sidebar.file_uploader(
        "2. Carregue os 4 arquivos CSV BRUTOS das estações:", 
        type="csv", 
        accept_multiple_files=True,
        help=f"Os nomes dos arquivos devem ser: {', '.join(estacoes_info.keys())}"
    )

    if st.sidebar.button("Gerar Previsão", type="primary"):
        if dados_brutos_files and len(dados_brutos_files) == len(estacoes_info):
            dados_historicos_processados = processar_arquivos_brutos(dados_brutos_files, estacoes_info)
            
            if dados_historicos_processados is not None:
                st.session_state.dados_historicos = dados_historicos_processados
                
                if len(dados_historicos_processados) < N_PAST:
                    st.error(f"Erro: Os dados resultaram em menos de {N_PAST} dias. Verifique os arquivos.")
                else:
                    modelo_ativo, scaler_ativo = modelos[cenario_cod], scalers[cenario_cod]
                    data_final = dados_historicos_processados.index.max()
                    dias_para_prever = (pd.to_datetime(f"{data_final.year}-12-31") - data_final).days
                    
                    if dias_para_prever > 0:
                        previsoes_vazao = fazer_previsao_futura(dados_historicos_processados, modelo_ativo, scaler_ativo, dias_para_prever)
                        datas_previsao = pd.date_range(start=data_final + timedelta(days=1), periods=dias_para_prever)
                        colunas_vazao = [col for col in scaler_ativo.feature_names_in_ if 'vazao' in col]
                        st.session_state.df_previsoes = pd.DataFrame(previsoes_vazao, index=datas_previsao, columns=colunas_vazao)
                    else:
                        st.warning("Os dados fornecidos já cobrem todo o ano.")
                        if 'df_previsoes' in st.session_state: del st.session_state['df_previsoes']
        else:
            st.sidebar.warning(f"Por favor, carregue os {len(estacoes_info)} arquivos CSV das estações.")

    # --- ÁREA DE RESULTADOS ---
    if 'df_previsoes' in st.session_state and 'dados_historicos' in st.session_state:
        st.header("📈 Gráfico de Previsões de Vazão")
        df_previsoes = st.session_state.df_previsoes
        dados_historicos = st.session_state.dados_historicos
        
        fig = go.Figure()
        cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, col in enumerate(df_previsoes.columns):
            id_estacao_limpo = col.replace('vazao_', '').replace('_', ' ').title()
            fig.add_trace(go.Scatter(x=dados_historicos.index, y=dados_historicos[col], mode='lines', name=f'Histórico {id_estacao_limpo}', line=dict(color=cores[i%len(cores)])))
            fig.add_trace(go.Scatter(x=df_previsoes.index, y=df_previsoes[col], mode='lines', name=f'Previsão {id_estacao_limpo}', line=dict(color=cores[i%len(cores)], dash='dash')))

        fig.update_layout(title_text="Vazão Histórica e Prevista para as Estações", xaxis_title="Data", yaxis_title="Vazão (m³/s)")
        st.plotly_chart(fig, use_container_width=True)
        
        csv_data = convert_df_to_csv(df_previsoes)
        st.download_button(
           label="📥 Baixar Dados da Previsão (CSV)",
           data=csv_data,
           file_name=f"previsao_vazao_{cenario_cod}.csv",
           mime="text/csv",
        )
        
        # =========================================================================== #
        # !!! INTERFACE FINAL CORRIGIDA, COM LIMITES DINÂMICOS !!!                  #
        # =========================================================================== #
        st.sidebar.header("Etapa 2: Análise Específica")
        
        # Pega a primeira e a última distância para definir os limites do slider
        distancias = sorted([info['distancia_km'] for info in estacoes_info.values()])
        dist_min, dist_max = min(distancias), max(distancias)
        
        data_min, data_max = df_previsoes.index.min().date(), df_previsoes.index.max().date()
        
        data_consulta = st.sidebar.date_input("1. Selecione uma data futura:", min_value=data_min, max_value=data_max, value=data_min)
        
        distancia_consulta = st.sidebar.slider(
            "2. Selecione a distância do Açude (km):", 
            min_value=dist_min, 
            max_value=dist_max, 
            value=dist_min, # Começa no valor mínimo
            step=0.5
        )
        
        st.sidebar.caption(f"Nota: A previsão só é válida entre {dist_min} km e {dist_max} km, que são os limites das estações.")
        
        if st.sidebar.button("Consultar Vazão Específica"):
            # Usa a função de interpolação segura
            funcao_calculo = interpolar_vazao(pd.to_datetime(data_consulta), df_previsoes, estacoes_info)
            vazao_calculada = funcao_calculo(distancia_consulta)
            st.sidebar.metric(label=f"Vazão em {data_consulta.strftime('%d/%m/%Y')} a {distancia_consulta} km", value=f"{vazao_calculada:.2f} m³/s")