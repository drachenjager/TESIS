# models/dl_models.py
import numpy as np, math, pandas as pd, torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def prepare_data_dl(series, lag=1):
    df = pd.DataFrame(series, columns=['y'])
    df['x_lag'] = df['y'].shift(lag)
    df.dropna(inplace=True)
    X = df[['x_lag']].values.reshape(-1, 1, 1)
    y = df['y'].values
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

class RNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc  = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_model(model_cls, train_data, test_data,
                epochs=50, batch_size=8, steps=1):
    # Si train_data muy pequeño, fallback de persistencia
    if len(train_data) <= 1:
        preds = np.repeat(train_data.iloc[-1], len(test_data))
        mae   = np.mean(np.abs(preds - test_data))
        rmse  = math.sqrt(np.mean((preds - test_data) ** 2))
        fc_vals = [float(train_data.iloc[-1])] * steps
        return mae, rmse, preds, fc_vals

    X_train, y_train = prepare_data_dl(train_data)
    if X_train.shape[0] == 0:
        preds = np.repeat(train_data.iloc[-1], len(test_data))
        mae   = np.mean(np.abs(preds - test_data))
        rmse  = math.sqrt(np.mean((preds - test_data) ** 2))
        fc_vals = [float(train_data.iloc[-1])] * steps
        return mae, rmse, preds, fc_vals

    loader = DataLoader(TensorDataset(X_train, y_train),
                        batch_size=batch_size, shuffle=False)

    model = model_cls()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb).squeeze(), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    preds = []
    cur = X_train[-1:].clone()                 # (1,1,1)
    for val in test_data:
        with torch.no_grad():
            y_pred = model(cur)
        preds.append(y_pred.item())
        cur = torch.tensor([[[float(val)]]], dtype=torch.float32)

    preds = np.array(preds)
    t     = np.asarray(test_data)
    mae   = np.mean(np.abs(preds - t))
    rmse  = math.sqrt(np.mean((preds - t) ** 2))

    next_val_in = torch.tensor([[[float(test_data.iloc[-1])]]], dtype=torch.float32)
    fc_vals = []
    for _ in range(steps):
        with torch.no_grad():
            nxt = model(next_val_in).item()
        fc_vals.append(float(nxt))
        next_val_in = torch.tensor([[[float(nxt)]]], dtype=torch.float32)

    return mae, rmse, preds, fc_vals

def train_rnn(train_data, test_data, steps=1):
    mae, rmse, preds, fc = train_model(RNNModel, train_data, test_data, steps=steps)
    return {"Modelo":"RNN","MAE":round(mae,4),"RMSE":round(rmse,4)}, preds, fc

def train_lstm(train_data, test_data, steps=1):
    mae, rmse, preds, fc = train_model(LSTMModel, train_data, test_data, steps=steps)
    return {"Modelo":"LSTM","MAE":round(mae,4),"RMSE":round(rmse,4)}, preds, fc

