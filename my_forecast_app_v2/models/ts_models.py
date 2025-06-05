# models/ts_models.py
import numpy as np, math
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing

def _mae_rmse(preds, truth):
    p = np.asarray(preds); t = np.asarray(truth)
    mae  = np.mean(np.abs(p - t))
    rmse = math.sqrt(np.mean((p - t) ** 2))
    return mae, rmse

def train_sarima(train_data, test_data, steps=1):
    model = SARIMAX(train_data, order=(1,1,1),
                    seasonal_order=(1,1,1,12),
                    enforce_stationarity=False,
                    enforce_invertibility=False)
    fit = model.fit(disp=False)

    preds = fit.predict(start=len(train_data),
                        end=len(train_data)+len(test_data)-1)
    mae, rmse = _mae_rmse(preds, test_data)
    fc_next   = np.asarray(fit.forecast(steps=steps))

    return {"Modelo":"SARIMA","MAE":round(mae,4),"RMSE":round(rmse,4)}, \
           np.asarray(preds), fc_next

def train_holtwinters(train_data, test_data, steps=1):
    if len(train_data) < 24:
        model = ExponentialSmoothing(train_data, trend='add', seasonal=None)
    else:
        model = ExponentialSmoothing(train_data, trend='add',
                                     seasonal='mul', seasonal_periods=12)
    fit = model.fit()

    preds = fit.predict(start=len(train_data),
                        end=len(train_data)+len(test_data)-1)
    mae, rmse = _mae_rmse(preds, test_data)
    fc_next   = np.asarray(fit.forecast(steps=steps))

    return {"Modelo":"Holt‑Winters","MAE":round(mae,4),"RMSE":round(rmse,4)}, \
           np.asarray(preds), fc_next

