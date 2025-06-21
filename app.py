import streamlit as st
import torch
from src.model import MultiLSTM
from src.agent import ActorCritic, train_ac
from src.utils import load_data, TimeSeriesEnv

st.title("Multivariate LSTM + Actor‑Critic Forecasting App")

data = load_data("data/timeseries_multivariate.csv")
seq_len = 20
input_size = data.shape[1] - 1

model = MultiLSTM(input_size=input_size)
agent = ActorCritic(state_dim=input_size * seq_len)

env = TimeSeriesEnv(data, seq_len, model)

if st.button("Train Agent"):
    train_ac(agent, env, episodes=100, gamma=0.95)
    st.success("Training complete!")

st.write("🔹 Use the trained agent to simulate actions…")
