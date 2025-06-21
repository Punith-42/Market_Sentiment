import streamlit as st
import numpy as np
import joblib

model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')
le_class = joblib.load('label_encoder_class.pkl')
le_coin = joblib.load('label_encoder_coin.pkl')
le_side = joblib.load('label_encoder_side.pkl')

st.set_page_config(page_title="Market Sentiment Predictor", layout="centered")
st.title(" Market Sentiment Predictor")

st.markdown("Enter trade details below to predict **market sentiment**.")

size_usd = st.number_input(" Size USD", min_value=0.0, value=10000.0)
closed_pnl = st.number_input(" Closed PnL", value=100.0)
start_position = st.number_input(" Start Position", value=1000.0, min_value=1.0)

coin = st.selectbox(" Coin", le_coin.classes_)
side = st.selectbox(" Side", le_side.classes_)


coin_encoded = le_coin.transform([coin])[0]
side_encoded = le_side.transform([side])[0]


leverage_proxy = size_usd / start_position if start_position != 0 else 0


sample = np.array([[size_usd, closed_pnl, leverage_proxy, coin_encoded, side_encoded]])
sample_scaled = scaler.transform(sample)


if st.button("🔍 Predict Market Sentiment"):
    pred_encoded = model.predict(sample_scaled)
    sentiment = le_class.inverse_transform(pred_encoded)[0]

    st.success(f"Predicted Sentiment: **{sentiment}**")


    prob = model.predict_proba(sample_scaled).max()
    st.write(f"🧪 Prediction Confidence: `{round(prob * 100, 2)}%`")