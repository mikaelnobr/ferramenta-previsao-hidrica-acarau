import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
import os
from tensorflow.keras.models import load_model # type: ignore
import plotly.graph_objects as go
from datetime import timedelta

st.set_page_config(page_title="Gestão Hídrica Acaraú", layout="wide")

@st.cache_resource
def carregar_tudo():
    try:
        with open('config.json', 'r', encoding='utf-8') as f: config = json.load(f)
    except FileNotFoundError:
        st.error("ERRO CRÍTICO: 'config.json' não foi encontrado."); return None, None
    
    modelos, scalers = {}, {}
    for cenario in ['baixo', 'medio', 'alto']:
        if os.path.exists(f'models/modelo_{cenario}.h5'):
            modelos[cenario] = load_model(f'models/modelo_{cenario}.h5')
            scalers[cenario] = joblib.load(f'models/scaler_{cenario}.pkl')
            
    return config, modelos, scalers

config, modelos, scalers = carregar_tudo()
if config: estacoes_info = config.get('estacoes', {})
N_PAST = 30 

def processar_arquivos_brutos(arquivos_carregados, estacoes_info):
    with st.spinner("Processando dados..."):
        lista_dfs = []
        mapa_arquivos = {f.name: f for f in arquivos_carregados}
        for nome_arq, info in estacoes_info.items():
            if nome_arq not in mapa_arquivos:
                st.error(f"Arquivo não encontrado: '{nome_arq}'."); return None
            arquivo = mapa_arquivos[nome_arq]
            try:
                df = pd.read_csv(arquivo, encoding='latin-1', sep=';', thousands=',')
                if 'Data' not in df.columns: df = pd.read_csv(arquivo, encoding='latin-1', sep=',', thousands=',')
            except Exception as e:
                st.error(f"Erro ao ler '{arquivo.name}': {e}"); return None
            id_estacao = nome_arq.replace('.csv', '').lower().replace(' ', '_').replace('-', '_')
            df = df.rename(columns={'Chuva (mm)': f'chuva_{id_estacao}', 'Nível (cm)': f'nivel_{id_estacao}', 'Vazão (m3/s)': f'vazao_{id_estacao}'})
            df['timestamp'] = pd.to_datetime(df['Data'] + ' ' + df['Hora'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
            df.dropna(subset=['timestamp'], inplace=True)
            df = df.set_index('timestamp')
            cols_renomeadas = [c for c in df.columns if any(s in c for s in ['chuva', 'nivel', 'vazao'])]
            lista_dfs.append(df[cols_renomeadas])
        df_consolidado = pd.concat(lista_dfs, axis=1)
        agg_funcs = {}
        for nome_arq, info in estacoes_info.items():
            id_estacao = nome_arq.replace('.csv', '').lower().replace(' ', '_').replace('-', '_')
            for tipo in ['chuva', 'nivel', 'vazao']:
                col = f'{tipo}_{id_estacao}'
                if col in df_consolidado.columns:
                    df_consolidado[col] = pd.to_numeric(df_consolidado[col], errors='coerce')
                    df_consolidado[col] = df_consolidado[col].interpolate(method='linear') if tipo != 'chuva' else df_consolidado[col].fillna(0)
                    agg_funcs[col] = 'sum' if tipo == 'chuva' else 'mean'
        df_diario = df_consolidado.resample('D').agg(agg_funcs)
        for col in df_diario.columns:
            if 'chuva' not in col: df_diario[col] = df_diario[col].interpolate(method='linear')
        df_diario.fillna(0, inplace=True)
        dia_do_ano = df_diario.index.dayofyear
        df_diario['dia_sin'] = np.sin(2 * np.pi * dia_do_ano / 366.0)
        df_diario['dia_cos'] = np.cos(2 * np.pi * dia_do_ano / 366.0)
        for nome_arq, info in estacoes_info.items():
            id_estacao = nome_arq.replace('.csv', '').lower().replace(' ', '_').replace('-', '_')
            df_diario[f'distancia_{id_estacao}'] = info['distancia_km']
        return df_diario

def fazer_previsao_futura(df_input_real, modelo, scaler, dias_para_prever):
    ordem_colunas = scaler.feature_names_in_
    df_transformed = df_input_real.copy()

    # PASSO 1: Transforma os dados de entrada para a escala LOG, assim como no treino
    vazao_cols = [col for col in df_transformed.columns if 'vazao' in col]
    for col in vazao_cols:
        df_transformed[col] = np.log1p(df_transformed[col])
    
    df_transformed = df_transformed.reindex(columns=ordem_colunas)
    dados_normalizados = scaler.transform(df_transformed)
    input_atual = dados_normalizados[-N_PAST:].reshape(1, N_PAST, dados_normalizados.shape[1])
    previsoes_normalizadas = []
    progresso = st.progress(0, text="Calculando previsões...")
    datas_previsao = pd.date_range(start=df_input_real.index.max() + timedelta(days=1), periods=dias_para_prever)
    
    for i in range(dias_para_prever):
        previsao_completa = modelo.predict(input_atual, verbose=0)
        previsoes_normalizadas.append(previsao_completa[0])
        novo_registro = input_atual[0, -1, :].copy()
        dia_futuro = datas_previsao[i].dayofyear
        novo_registro[list(ordem_colunas).index('dia_sin')] = np.sin(2 * np.pi * dia_futuro / 366.0)
        novo_registro[list(ordem_colunas).index('dia_cos')] = np.cos(2 * np.pi * dia_futuro / 366.0)
        target_indices = [list(ordem_colunas).index(c) for c in ordem_colunas if any(s in c for s in ['chuva', 'nivel', 'vazao'])]
        for j, idx in enumerate(target_indices):
            novo_registro[idx] = previsao_completa[0][j]
        input_atual = np.append(input_atual[:, 1:, :], [[novo_registro]], axis=1)
        progresso.progress((i + 1) / dias_para_prever, text=f"Calculando dia {i+1}/{dias_para_prever}")
        
    progresso.empty()
    previsoes_normalizadas = np.array(previsoes_normalizadas)

    # PASSO 2: Desnormaliza para a escala LOG
    dummy_array = np.zeros((dias_para_prever, len(ordem_colunas)))
    target_indices_previsao = [list(ordem_colunas).index(c) for c in ordem_colunas if any(s in c for s in ['chuva', 'nivel', 'vazao'])]
    for j, idx in enumerate(target_indices_previsao):
        dummy_array[:, idx] = previsoes_normalizadas[:, j]
    previsoes_desnormalizadas_log = scaler.inverse_transform(dummy_array)
    
    # PASSO 3: Cria um DataFrame temporário na escala LOG
    df_previsoes_log = pd.DataFrame(previsoes_desnormalizadas_log, columns=ordem_colunas, index=datas_previsao)
    
    # PASSO 4: Reverte a transformação LOG para obter os valores REAIS
    df_previsoes_real = df_previsoes_log.copy()
    for col in vazao_cols:
        df_previsoes_real[col] = np.expm1(df_previsoes_log[col])
    
    df_previsoes_real.clip(lower=0, inplace=True)
    return df_previsoes_real

# ... O resto do seu código (interpolar_vazao, convert_df_to_csv, e a interface) permanece o mesmo ...
def interpolar_vazao(data_alvo, previsoes_df, estacoes_info):
    distancias = np.array([info['distancia_km'] for info in estacoes_info.values()])
    vazoes_previstas = [f'vazao_{k.replace(".csv", "").lower().replace(" ", "_").replace("-", "_")}' for k in estacoes_info.keys()]
    vazoes = previsoes_df.loc[data_alvo, vazoes_previstas].values
    idx_sorted = np.argsort(distancias)
    return lambda dist_km: np.interp(dist_km, distancias[idx_sorted], vazoes[idx_sorted])

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=True, index_label='Data', sep=';', decimal=',').encode('utf-8-sig')

st.title("💧 Ferramenta Preditiva para Gestão Hídrica - Vale do Acaraú")
if not modelos:
    st.error("Nenhum modelo treinado foi encontrado.")
else:
    st.sidebar.header("Etapa 1: Previsão Geral")
    mapa_cenarios = {'Baixa Precipitação': 'baixo', 'Média Precipitação': 'medio', 'Alta Precipitação': 'alto'}
    cenarios_disp = [k for k, v in mapa_cenarios.items() if v in modelos]
    cenario_display = st.sidebar.selectbox("1. Selecione o cenário:", cenarios_disp)
    cenario_cod = mapa_cenarios[cenario_display]
    dados_brutos = st.sidebar.file_uploader("2. Carregue os 4 arquivos CSV BRUTOS:", type="csv", accept_multiple_files=True, help=f"Nomes: {', '.join(estacoes_info.keys())}")
    if st.sidebar.button("Gerar Previsão", type="primary"):
        if dados_brutos and len(dados_brutos) == len(estacoes_info):
            dados_hist = processar_arquivos_brutos(dados_brutos, estacoes_info)
            if dados_hist is not None:
                st.session_state.dados_historicos = dados_hist
                if len(dados_hist) < N_PAST:
                    st.error(f"Erro: Dados insuficientes (< {N_PAST} dias).")
                else:
                    modelo, scaler = modelos[cenario_cod], scalers[cenario_cod]
                    data_final = dados_hist.index.max()
                    dias_a_prever = (pd.to_datetime(f"{data_final.year}-12-31") - data_final).days
                    if dias_a_prever > 0:
                        st.session_state.df_previsoes = fazer_previsao_futura(dados_hist, modelo, scaler, dias_a_prever)
                    else:
                        st.warning("Dados já cobrem todo o ano.")
                        if 'df_previsoes' in st.session_state: del st.session_state['df_previsoes']
        else:
            st.sidebar.warning(f"Por favor, carregue os {len(estacoes_info)} arquivos.")
    if 'df_previsoes' in st.session_state:
        st.header("📈 Gráfico de Previsões de Vazão")
        df_previsoes, dados_hist = st.session_state.df_previsoes, st.session_state.dados_historicos
        vazao_cols = [c for c in df_previsoes.columns if 'vazao' in c]
        fig = go.Figure()
        cores = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, col in enumerate(vazao_cols):
            nome_limpo = col.replace('vazao_', '').replace('_', ' ').title()
            fig.add_trace(go.Scatter(x=dados_hist.index, y=dados_hist[col], mode='lines', name=f'Histórico {nome_limpo}', line=dict(color=cores[i%len(cores)])))
            fig.add_trace(go.Scatter(x=df_previsoes.index, y=df_previsoes[col], mode='lines', name=f'Previsão {nome_limpo}', line=dict(color=cores[i%len(cores)], dash='dash')))
        fig.update_layout(title_text="Vazão Histórica e Prevista", xaxis_title="Data", yaxis_title="Vazão (m³/s)")
        st.plotly_chart(fig, use_container_width=True)
        csv_data = convert_df_to_csv(df_previsoes[vazao_cols])
        st.download_button(label="📥 Baixar Previsão (CSV)", data=csv_data, file_name=f"previsao_{cenario_cod}.csv", mime="text/csv")
        st.sidebar.header("Etapa 2: Análise Específica")
        distancias = sorted([info['distancia_km'] for info in estacoes_info.values()])
        dist_min, dist_max = min(distancias), max(distancias)
        data_min, data_max = df_previsoes.index.min().date(), df_previsoes.index.max().date()
        data_consulta = st.sidebar.date_input("1. Data da consulta:", min_value=data_min, max_value=data_max, value=data_min)
        dist_consulta = st.sidebar.slider("2. Distância do Açude (km):", min_value=dist_min, max_value=dist_max, value=dist_min, step=0.5)
        st.sidebar.caption(f"Previsão válida entre {dist_min} km e {dist_max} km.")
        if st.sidebar.button("Consultar Vazão"):
            vazao_calc = interpolar_vazao(pd.to_datetime(data_consulta), df_previsoes, estacoes_info)(dist_consulta)
            st.sidebar.metric(label=f"Vazão em {data_consulta.strftime('%d/%m/%Y')} a {dist_consulta} km", value=f"{vazao_calc:.2f} m³/s")