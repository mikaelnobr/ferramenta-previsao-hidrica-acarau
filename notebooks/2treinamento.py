import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore

print("--- FASE 2: INICIANDO TREINAMENTO DOS MODELOS ---")

# 1. Função de Janelamento (prepara os dados para a LSTM)
def create_sequences(data, n_past, n_future, target_indices):
    """Cria 'janelas' de dados para o modelo aprender."""
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, :])
        y.append(data[i + n_future - 1, target_indices])
    return np.array(X), np.array(y).squeeze()

# 2. Configurações
CENARIOS = ['baixo', 'medio', 'alto']
PROCESSED_DATA_PATH = os.path.join('data', 'processed')
MODELS_PATH = 'models/'
os.makedirs(MODELS_PATH, exist_ok=True)

# Hiperparâmetros do Modelo
N_PAST = 30         # Usar dados dos últimos 30 dias...
N_FUTURE = 1        # ...para prever o próximo 1 dia.
N_EPOCHS = 100      # Número máximo de rodadas de treinamento.
BATCH_SIZE = 32

# 3. Loop de Treinamento
for cenario in CENARIOS:
    print(f"\n--- Treinando Modelo para Cenário: {cenario.upper()} ---")
    
    cenario_file = os.path.join(PROCESSED_DATA_PATH, f'{cenario}_consolidado_diario.csv')
    if not os.path.exists(cenario_file):
        print(f"  AVISO: Arquivo processado '{cenario_file}' não encontrado. Pulando treinamento.")
        continue
        
    df = pd.read_csv(cenario_file, index_col=0, parse_dates=True).dropna()
    
    # Verifica se há dados suficientes para criar pelo menos uma janela de treinamento
    if len(df) < N_PAST + N_FUTURE:
        print(f"  AVISO: Dados insuficientes para o cenário '{cenario}' ({len(df)} linhas). São necessárias pelo menos {N_PAST + N_FUTURE} linhas. Pulando.")
        continue

    # Identifica automaticamente as colunas de vazão para serem o alvo da previsão
    target_cols = [col for col in df.columns if 'vazao' in col]
    target_indices = [df.columns.get_loc(col) for col in target_cols]
    
    # Normaliza os dados (coloca tudo entre 0 e 1) e salva o "normalizador"
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df)
    joblib.dump(scaler, os.path.join(MODELS_PATH, f'scaler_{cenario}.pkl'))
    print(f"  Scaler salvo em 'models/scaler_{cenario}.pkl'")
    
    # Cria as "janelas" de dados de treino e alvo
    X, y = create_sequences(data_scaled, N_PAST, N_FUTURE, target_indices)
    
    # Constrói a arquitetura da Rede Neural LSTM
    model = Sequential([
        LSTM(units=100, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
        Dropout(0.2),
        LSTM(units=50),
        Dropout(0.2),
        Dense(units=len(target_cols)) # A camada de saída tem um neurônio para cada vazão a ser prevista
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # Define uma parada automática para o treinamento quando ele parar de melhorar
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    # Inicia o treinamento
    print("\n  Iniciando o treinamento do modelo...")
    model.fit(
        X, y,
        epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2, # Separa 20% dos dados para validar o modelo a cada rodada
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Salva o modelo treinado
    model.save(os.path.join(MODELS_PATH, f'modelo_{cenario}.h5'))
    print(f"  --> Modelo salvo com sucesso em 'models/modelo_{cenario}.h5'")

print("\n--- FASE 2 CONCLUÍDA! Modelos estão na pasta 'models'. ---")