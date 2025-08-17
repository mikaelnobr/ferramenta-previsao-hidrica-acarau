import pandas as pd
import numpy as np
import os
import keras_tuner as kt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import LSTM, Dense, Dropout # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore

print("--- FASE DE TUNING: ENCONTRANDO OS MELHORES HIPERPARÂMETROS ---")

def create_sequences(data, n_past, n_future, target_indices):
    X, y = [], []
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, :])
        y.append(data[i + n_future - 1, target_indices])
    return np.array(X), np.array(y).squeeze()

# Escolha o cenário com mais dados ou mais representativo para o tuning
# Geralmente, o cenário de 'alta' precipitação tem a dinâmica mais complexa.
CENARIO_PARA_TUNING = 'alto' 

PROCESSED_DATA_PATH = os.path.join('data', 'processed')
df = pd.read_csv(os.path.join(PROCESSED_DATA_PATH, f'{CENARIO_PARA_TUNING}_consolidado_diario.csv'), index_col=0, parse_dates=True).dropna()

target_cols = [col for col in df.columns if 'chuva' in col or 'nivel' in col or 'vazao' in col]
target_indices = [df.columns.get_loc(col) for col in target_cols]

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df)

N_PAST = 30
X, y = create_sequences(data_scaled, N_PAST, 1, target_indices)

# Função que constrói o modelo para o tuner testar
def build_model(hp):
    model = Sequential()
    
    model.add(LSTM(
        units=hp.Int('units_1', min_value=32, max_value=128, step=32),
        return_sequences=True,
        input_shape=(X.shape[1], X.shape[2])
    ))
    model.add(Dropout(rate=hp.Float('dropout_1', min_value=0.1, max_value=0.4, step=0.1)))
    
    model.add(LSTM(
        units=hp.Int('units_2', min_value=16, max_value=96, step=32)
    ))
    model.add(Dropout(rate=hp.Float('dropout_2', min_value=0.1, max_value=0.4, step=0.1)))
    
    model.add(Dense(units=len(target_cols)))
    
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mean_squared_error'
    )
    return model

tuner = kt.Hyperband(
    build_model,
    objective='val_loss',
    max_epochs=50,
    factor=3,
    directory='tuning_dir',
    project_name=f'tuning_{CENARIO_PARA_TUNING}'
)

print(f"\nIniciando a busca para o cenário '{CENARIO_PARA_TUNING}'. Isso pode demorar muito tempo...")
tuner.search(X, y, epochs=50, validation_split=0.2)

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
--- BUSCA COMPLETA ---
Os melhores hiperparâmetros encontrados são:

- Unidades na Camada 1: {best_hps.get('units_1')}
- Dropout na Camada 1: {best_hps.get('dropout_1'):.2f}
- Unidades na Camada 2: {best_hps.get('units_2')}
- Dropout na Camada 2: {best_hps.get('dropout_2'):.2f}
- Taxa de Aprendizado: {best_hps.get('learning_rate')}

Agora, atualize o script '2treinamento.py' com esses valores e execute-o.
""")