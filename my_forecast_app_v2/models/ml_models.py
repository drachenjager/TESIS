# models/ml_models.py
import numpy as np, math, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

LAG = 1

def create_supervised(series, lag=LAG):
    df = pd.DataFrame(series, columns=['y'])
    df['x_lag'] = df['y'].shift(lag)
    df.dropna(inplace=True)
    return df[['x_lag']].values, df['y'].values

def _mae_rmse(preds, truth):
    p = np.asarray(preds); t = np.asarray(truth)
    mae  = np.mean(np.abs(p - t))
    rmse = math.sqrt(np.mean((p - t) ** 2))
    return mae, rmse

def _safe_return(name, preds, fc, test_data):
    mae, rmse = _mae_rmse(preds, test_data)
    metrics = {
        "Modelo": name,
        "MAE":   round(mae,4)   if not np.isnan(mae)   else "–",
        "RMSE":  round(rmse,4)  if not np.isnan(rmse)  else "–"
    }
    return metrics, preds, fc

def train_linear_regression(train_data, test_data, steps=1):
    X, y = create_supervised(train_data)
    if X.shape[0] == 0:
        preds = np.repeat(train_data.iloc[-1], len(test_data))
        fc_vals = [float(train_data.iloc[-1])] * steps
        return _safe_return("Regresión Lineal", preds, fc_vals, test_data)

    model = LinearRegression().fit(X, y)
    preds, cur = [], train_data.iloc[-1]
    for val in test_data:
        preds.append(model.predict([[cur]])[0])
        cur = val
    preds = np.array(preds)
    fc_vals = []
    cur = test_data.iloc[-1]
    for _ in range(steps):
        nxt = model.predict([[cur]])[0]
        fc_vals.append(float(nxt))
        cur = nxt
    return _safe_return("Regresión Lineal", preds, fc_vals, test_data)

def train_random_forest(train_data, test_data, steps=1):
    X, y = create_supervised(train_data)
    if X.shape[0] == 0:
        preds = np.repeat(train_data.iloc[-1], len(test_data))
        fc_vals = [float(train_data.iloc[-1])] * steps
        return _safe_return("Random Forest", preds, fc_vals, test_data)

    model = RandomForestRegressor(n_estimators=100).fit(X, y)
    preds, cur = [], train_data.iloc[-1]
    for val in test_data:
        preds.append(model.predict([[cur]])[0])
        cur = val
    preds = np.array(preds)
    fc_vals = []
    cur = test_data.iloc[-1]
    for _ in range(steps):
        nxt = model.predict([[cur]])[0]
        fc_vals.append(float(nxt))
        cur = nxt
    return _safe_return("Random Forest", preds, fc_vals, test_data)

