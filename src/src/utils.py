import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class TimeSeriesEnv:
    def __init__(self, data, seq_len, model):
        self.data = data
        self.seq_len = seq_len
        self.model = model
        self.scaler = MinMaxScaler()
        self.idx = seq_len

        self.scaled = self.scaler.fit_transform(data)
        self.max_idx = len(data) - 1

    def reset(self):
        self.idx = self.seq_len
        return self._get_state()

    def _get_state(self):
        seq = self.scaled[self.idx - self.seq_len : self.idx]
        return seq.flatten()

    def step(self, action):
        pred = self.model(torch.zeros((1, self.seq_len, self.data.shape[1]-1)))
        current = self.scaled[self.idx, 0]
        next_val = self.scaled[self.idx + 1, 0]
        reward = (next_val - current) if action == 1 else \
                 -(next_val - current) if action == 2 else 0
        self.idx += 1
        done = self.idx >= self.max_idx
        return self._get_state(), reward, done

def load_data(path):
    return pd.read_csv(path).values
