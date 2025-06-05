# app.py
from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from models.ts_models import train_sarima, train_holtwinters
from models.ml_models import train_linear_regression, train_random_forest
from models.dl_models import train_rnn, train_lstm

warnings.filterwarnings("ignore", category=ConvergenceWarning)

app = Flask(__name__)

def set_internal_freq(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    df.index = pd.to_datetime(df.index)
    if interval == "1d":
        return df.asfreq("B").ffill()
    if interval == "1wk":
        return df.asfreq("W").ffill()
    if interval == "1mo":
        return df.asfreq("MS").ffill()
    return df

@app.route("/", methods=["GET", "POST"])
def index():
    tables_html, forecast_values, error_msg = None, None, None
    horizon_select = 1

    if request.method == "POST":
        period_select   = request.form.get("period_select", "6mo")
        interval_select = request.form.get("interval_select", "1d")
        horizon_select  = int(request.form.get("horizon_select", "1"))

        try:
            df_raw = yf.download("MXN=X",
                                 period=period_select,
                                 interval=interval_select,
                                 auto_adjust=True)
            if df_raw.empty:
                raise ValueError("No se encontraron datos para ese rango.")

            df = df_raw[['Close']].rename(columns={'Close':'y'})
            df = set_internal_freq(df, interval_select)

            total_points = len(df)
            if total_points < 3:
                raise ValueError("Menos de 3 observaciones. Amplía el periodo.")

            test_size  = min(12, max(1, total_points // 4))
            train_data = df['y'][:-test_size]
            test_data  = df['y'][-test_size:]
            if len(train_data) < 3:
                raise ValueError("Entrenamiento insuficiente (min 3 filas).")

            sarima_m, _, sarima_fc = train_sarima(train_data, test_data, horizon_select)
            holt_m,  _, holt_fc   = train_holtwinters(train_data, test_data, horizon_select)
            lr_m,    _, lr_fc     = train_linear_regression(train_data, test_data, horizon_select)
            rf_m,    _, rf_fc     = train_random_forest(train_data, test_data, horizon_select)
            rnn_m,   _, rnn_fc    = train_rnn(train_data, test_data, horizon_select)
            lstm_m,  _, lstm_fc   = train_lstm(train_data, test_data, horizon_select)

            metrics_df = pd.DataFrame([sarima_m, holt_m, lr_m, rf_m, rnn_m, lstm_m])
            tables_html = metrics_df.to_html(index=False, classes="table table-striped")

            forecast_values = {
                "SARIMA":       list(sarima_fc),
                "Holt‑Winters": list(holt_fc),
                "LinearReg":    list(lr_fc),
                "RandomForest": list(rf_fc),
                "RNN":          list(rnn_fc),
                "LSTM":         list(lstm_fc)
            }

        except Exception as exc:
            error_msg = str(exc)

    return render_template("index.html",
                           tables=tables_html,
                           forecast_values=forecast_values,
                           error_msg=error_msg,
                           horizon=horizon_select if request.method == "POST" else None)

if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
