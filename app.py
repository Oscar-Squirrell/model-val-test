from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import jarque_bera
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


app = FastAPI(title="Model Validation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # allow Caffeine drafts
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataInput(BaseModel):
    columns: list[str]
    data: list[list[float]]

@app.post("/run-tests")
def run_tests(payload: DataInput):
    # Convert to DataFrame
    df = pd.DataFrame(payload.data, columns=payload.columns)
    results = {}

    # Basic stats
    results["mean"] = df.mean().to_dict()
    results["std"] = df.std().to_dict()
    results["min"] = df.min().to_dict()
    results["max"] = df.max().to_dict()

    # Durbin-Watson
    try:
        results["durbin_watson"] = sm.stats.durbin_watson(df.iloc[:,0])
    except Exception as e:
        results["durbin_watson"] = f"error: {e}"

    # Jarque–Bera test for normality
    try:
        jb_stat, jb_p = jarque_bera(df.iloc[:,0])
        results["jarque_bera"] = {"statistic": float(jb_stat), "p_value": float(jb_p)}
    except Exception as e:
        results["jarque_bera"] = f"error: {e}"
    #Time Series Diagnostics
    series = df.iloc[:, 0].dropna()
    ts = {}

    # ADF test
    try:
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(series)
        ts["adf"] = {
            "statistic": float(adf_stat),
            "p_value": float(adf_p),
            "lags": int(adf_lags),
            "nobs": int(adf_nobs),
            "critical_values": {k: float(v) for k, v in adf_crit.items()}
        }
    except Exception as e:
        ts["adf"] = f"error: {e}"

    # KPSS test
    try:
        kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(series, regression='c')
        ts["kpss"] = {
            "statistic": float(kpss_stat),
            "p_value": float(kpss_p),
            "lags": int(kpss_lags),
            "critical_values": {k: float(v) for k, v in kpss_crit.items()}
        }
    except Exception as e:
        ts["kpss"] = f"error: {e}"

    # ACF
    try:
        ts["acf"] = [float(v) for v in acf(series, fft=False)]
    except Exception as e:
        ts["acf"] = f"error: {e}"

    # PACF
    try:
        ts["pacf"] = [float(v) for v in pacf(series, method='ywunbiased')]
    except Exception as e:
        ts["pacf"] = f"error: {e}"

    # Ljung-Box
    try:
        lb_stat, lb_p = acorr_ljungbox(series, lags=[10], return_df=False)
        ts["ljung_box"] = {
            "statistic": float(lb_stat[0]),
            "p_value": float(lb_p[0])
        }
    except Exception as e:
        ts["ljung_box"] = f"error: {e}"

    results["time_series"] = ts

    # Placeholder for future tests
    results["heteroskedasticity"] = {"statistic": None, "p_value": None}

    return {"tests": results}
