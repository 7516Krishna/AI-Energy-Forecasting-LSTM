import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# -----------------------------
# CREATE OUTPUT FOLDER
# -----------------------------
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# 1. LOAD DATA
# -----------------------------
df = pd.read_csv("data/energy.csv")

df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

data = df[['energy']]

print("✅ Data Loaded")
print(data.head())

# -----------------------------
# 2. NORMALIZE DATA
# -----------------------------
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# -----------------------------
# 3. CREATE SEQUENCES
# -----------------------------
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 24
X, y = create_sequences(scaled_data, seq_length)

print("✅ Sequences Created:", X.shape)

# -----------------------------
# 4. TRAIN TEST SPLIT
# -----------------------------
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# -----------------------------
# 5. BUILD LSTM MODEL
# -----------------------------
model = Sequential([
    LSTM(50, input_shape=(X_train.shape[1], 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

print("✅ Model Built")

# -----------------------------
# 6. TRAIN MODEL
# -----------------------------
history = model.fit(X_train, y_train, epochs=10, batch_size=32)

# -----------------------------
# 7. PREDICTION
# -----------------------------
predictions = model.predict(X_test)

# Convert back to original scale
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test)

print("✅ Prediction Done")

# -----------------------------
# 8. VISUALIZATION
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label="Actual")
plt.plot(predictions, label="Predicted")
plt.legend()
plt.title("LSTM Energy Forecasting")

plt.savefig("outputs/lstm_result.png")
plt.show()

print("✅ Graph saved in outputs/")