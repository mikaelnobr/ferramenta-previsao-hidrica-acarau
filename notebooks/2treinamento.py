import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.callbacks import EarlyStopping # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras import regularizers # type: ignore

print("--- FASE 2: INICIANDO TREINAMENTO DEFINITIVO (COM TRANSFORMAÇÃO LOGARÍTMICA) ---")

def create_sequences(data, n_past, n_future, target_indices):
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, :])
        y.append(data[i + n_future - 1, target_indices])
    return np.array(X), np.array(y).squeeze()

CENARIOS = ['baixo', 'medio', 'alto']
PROCESSED_DATA_PATH = os.path.join('data', 'processed')
MODELS_PATH = 'models/'
os.makedirs(MODELS_PATH, exist_ok=True)

N_PAST = 30
N_EPOCHS = 150 
BATCH_SIZE = 32

BEST_UNITS_1, BEST_DROPOUT_1 = 64, 0.4
BEST_UNITS_2, BEST_DROPOUT_2 = 80, 0.2
BEST_LR = 0.0001

for cenario in CENARIOS:
    print(f"\n--- Treinando Modelo para Cenário: {cenario.upper()} ---")
    
    cenario_file = os.path.join(PROCESSED_DATA_PATH, f'{cenario}_consolidado_diario.csv')
    if not os.path.exists(cenario_file):
        print(f"  AVISO: Arquivo não encontrado. Pulando."); continue
        
    df = pd.read_csv(cenario_file, index_col=0, parse_dates=True).dropna()
    
    if len(df) < N_PAST + 1:
        print(f"  AVISO: Dados insuficientes. Pulando."); continue

    # =========================================================================== #
    # !!! MUDANÇA CRÍTICA: APLICANDO A TRANSFORMAÇÃO LOGARÍTMICA !!!              #
    # =========================================================================== #
    vazao_cols = [col for col in df.columns if 'vazao' in col]
    for col in vazao_cols:
        df[col] = np.log1p(df[col]) # log1p é o mesmo que log(x + 1)
    
    target_cols = [col for col in df.columns if 'chuva' in col or 'nivel' in col or 'vazao' in col]
    target_indices = [df.columns.get_loc(col) for col in target_cols]
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(df)
    joblib.dump(scaler, os.path.join(MODELS_PATH, f'scaler_{cenario}.pkl'))
    print(f"  Scaler (treinado com dados log) salvo.")
    
    X, y = create_sequences(data_scaled, N_PAST, 1, target_indices)
    
    model = Sequential([
        LSTM(units=BEST_UNITS_1, return_sequences=True, input_shape=(X.shape[1], X.shape[2]),
             kernel_regularizer=regularizers.l2(0.001)),
        Dropout(rate=BEST_DROPOUT_1),
        LSTM(units=BEST_UNITS_2, kernel_regularizer=regularizers.l2(0.001)),
        Dropout(rate=BEST_DROPOUT_2),
        Dense(units=len(target_cols))
    ])
    
    # Usamos o MSE padrão aqui, pois a penalidade agora está implícita na transformação log
    model.compile(optimizer=Adam(learning_rate=BEST_LR), loss='mean_squared_error')
    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    print("\n  Iniciando o treinamento do modelo final...")
    model.fit(X, y, epochs=N_EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2, callbacks=[early_stopping], verbose=1)
    
    model.save(os.path.join(MODELS_PATH, f'modelo_{cenario}.h5'))
    print(f"  --> Modelo DEFINITIVO salvo com sucesso.")

print("\n--- FASE 2 (COM TREINAMENTO DEFINITIVO) CONCLUÍDA! ---")