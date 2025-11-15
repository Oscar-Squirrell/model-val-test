from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.stats import jarque_bera

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

    # Placeholder for future tests
    results["heteroskedasticity"] = {"statistic": None, "p_value": None}

    return {"tests": results}
