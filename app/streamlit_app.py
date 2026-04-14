# -----------------------------
# FIX WARNINGS (IMPORTANT)
# -----------------------------
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# -----------------------------
# IMPORTS
# -----------------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Energy Dashboard", layout="wide")

# -----------------------------
# CUSTOM UI STYLE
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.card {
    padding: 20px;
    border-radius: 15px;
    background: #1c1f26;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.3);
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("⚡ AI-Powered Energy Consumption Forecasting")
st.markdown("Upload dataset → Train LSTM → Visualize Predictions")

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Model Settings")

seq_length = st.sidebar.slider("Sequence Length (hours)", 12, 72, 24)
epochs = st.sidebar.slider("Epochs", 1, 20, 5)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # -----------------------------
        # LOAD DATA
        # -----------------------------
        df = pd.read_csv(uploaded_file)

        if 'datetime' not in df.columns or 'energy' not in df.columns:
            st.error("❌ CSV must contain 'datetime' and 'energy' columns")
            st.stop()

        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)

        # -----------------------------
        # TOP DASHBOARD CARDS
        # -----------------------------
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("📊 Total Records", len(df))

        with col2:
            st.metric("⚡ Avg Energy", f"{df['energy'].mean():.2f}")

        with col3:
            st.metric("🔺 Max Energy", f"{df['energy'].max():.2f}")

        # -----------------------------
        # DATA PREVIEW
        # -----------------------------
        st.subheader("📊 Dataset Preview")
        st.dataframe(df.head())

        # -----------------------------
        # NORMALIZATION
        # -----------------------------
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df[['energy']])

        # -----------------------------
        # SEQUENCE CREATION
        # -----------------------------
        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(len(data) - seq_length):
                X.append(data[i:i+seq_length])
                y.append(data[i+seq_length])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled_data, seq_length)

        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        st.write(f"📦 Training Samples: {X_train.shape}")

        # -----------------------------
        # MODEL TRAINING
        # -----------------------------
        st.subheader("🤖 Training LSTM Model")

        with st.spinner("Training model... ⏳"):

            model = Sequential([
                Input(shape=(X_train.shape[1], 1)),
                LSTM(50),
                Dense(1)
            ])

            model.compile(optimizer='adam', loss='mse')

            model.fit(X_train, y_train, epochs=epochs, batch_size=32, verbose=0)

        st.success("✅ Model trained successfully!")

        # -----------------------------
        # PREDICTIONS
        # -----------------------------
        predictions = model.predict(X_test)

        predictions = scaler.inverse_transform(predictions)
        y_test_actual = scaler.inverse_transform(y_test)

        # -----------------------------
        # METRICS
        # -----------------------------
        rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))

        col4, col5 = st.columns(2)

        with col4:
            st.metric("📉 RMSE", f"{rmse:.2f}")

        with col5:
            st.metric("📊 Test Samples", len(y_test_actual))

        # -----------------------------
        # PLOTLY GRAPH (MODERN UI)
        # -----------------------------
        st.subheader("📈 Forecast Visualization")

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=y_test_actual.flatten(),
            mode='lines',
            name='Actual'
        ))

        fig.add_trace(go.Scatter(
            y=predictions.flatten(),
            mode='lines',
            name='Predicted'
        ))

        fig.update_layout(
            template="plotly_dark",
            height=500,
            title="Energy Forecast (LSTM)"
        )

        st.plotly_chart(fig, width='stretch')

        # -----------------------------
        # DOWNLOAD RESULTS
        # -----------------------------
        result_df = pd.DataFrame({
            "Actual": y_test_actual.flatten(),
            "Predicted": predictions.flatten()
        })

        csv = result_df.to_csv(index=False).encode('utf-8')

        st.download_button(
            "⬇ Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error(f"⚠️ Error: {e}")