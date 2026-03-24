import io
import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import StreamingResponse
from fastapi.security import HTTPBearer

# 🔐 Firebase
import firebase_admin
from firebase_admin import credentials, auth

# =============================================================================
# 🔥 INIT FIREBASE
# =============================================================================
cred = credentials.Certificate(r"c:\Users\Ahmed\Desktop\firebase_key.json")
firebase_admin.initialize_app(cred)

security = HTTPBearer()

# =============================================================================
# 🔐 VERIFY USER
# =============================================================================
def verify_user(token=Depends(security)):
    try:
        decoded_token = auth.verify_id_token(token.credentials)
        return decoded_token
    except:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# =============================================================================
# LOAD MODEL
# =============================================================================
try:
    bundle = joblib.load("model_bundle.pkl")

    best_model = bundle["model"]
    BEST_LAGS = bundle["lags"]
    BEST_ROLL = bundle["roll"]
    TRAINING_RESULTS = bundle["results"]

except:
    raise Exception("❌ model_bundle.pkl not found. Run train.py first.")

# =============================================================================
# FASTAPI APP
# =============================================================================

app = FastAPI(
    swagger_ui_parameters={
        "persistAuthorization": True
    },
    title="AI Demand Forecast API 🚀",
    description="Secure SaaS Forecasting API",
    version="3.0.0",
)

from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse

@app.get("/docs", include_in_schema=False)
async def custom_docs():
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="My API"
    ).body.decode("utf-8")

    custom_js = """
    <script>
    window.onload = function() {
        const token = localStorage.getItem("token");

        if (token) {
            const ui = window.ui;

            ui.preauthorizeApiKey("Bearer", "Bearer " + token);
        }
    }
    </script>
    """

    html = html.replace("</body>", custom_js + "</body>")
    return HTMLResponse(html)
# =============================================================================
# REQUIRED COLUMNS
# =============================================================================
REQUIRED_COLUMNS = {
    "month",
    "product_id",
    "number_of_product_purchases",
    "number_of_times_added_to_cart",
    "number_of_times_add_followed_by_purchase",
    "number_of_times_add_followed_by_no_purchase",
}

# =============================================================================
# HELPERS
# =============================================================================
def validate_and_load(contents: bytes) -> pd.DataFrame:
    try:
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"CSV error: {exc}")

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise HTTPException(400, f"Missing columns: {sorted(missing)}")

    df = df.drop_duplicates()

    df["month"] = pd.to_datetime(df["month"], errors="coerce")
    if df["month"].isna().any():
        raise HTTPException(400, "Invalid dates in 'month' column")

    return df.sort_values(["product_id", "month"]).reset_index(drop=True)


def engineer_features(df):
    df = df.copy()

    df["conversion_rate"] = (
        df["number_of_times_add_followed_by_purchase"]
        / df["number_of_times_added_to_cart"].replace(0, np.nan)
    )

    df["cart_drop_rate"] = (
        df["number_of_times_add_followed_by_no_purchase"]
        / df["number_of_times_added_to_cart"].replace(0, np.nan)
    )

    return df.drop(
        columns=[
            "number_of_times_add_followed_by_purchase",
            "number_of_times_add_followed_by_no_purchase",
        ]
    )

# =============================================================================
# FORECAST
# =============================================================================
def forecast(df: pd.DataFrame) -> pd.DataFrame:
    future_predictions = []
    last_month = df["month"].max()

    forecast_months = [
        last_month + pd.DateOffset(months=1),
        last_month + pd.DateOffset(months=2),
    ]

    for product in df["product_id"].unique():

        product_df = df[df["product_id"] == product]

        if len(product_df) < BEST_LAGS:
            continue

        hist = product_df.iloc[-BEST_LAGS:]
        lag_values = hist["number_of_product_purchases"].values.tolist()

        conv_rate = hist["conversion_rate"].iloc[-1]
        drop_rate = hist["cart_drop_rate"].iloc[-1]

        for step in range(2):

            row = {f"lag_{i+1}": lag_values[i] for i in range(BEST_LAGS)}
            row["product_id"] = product

            valid_lags = [v for v in lag_values[:BEST_ROLL] if not np.isnan(v)]

            row[f"rolling_mean_{BEST_ROLL}"] = (
                np.mean(valid_lags) if valid_lags else np.nan
            )

            row[f"rolling_std_{BEST_ROLL}"] = (
                np.std(valid_lags) if len(valid_lags) > 1 else np.nan
            )

            row["conversion_rate"] = conv_rate
            row["cart_drop_rate"] = drop_rate

            pred = max(
                0.0,
                round(float(best_model.predict(pd.DataFrame([row]))[0]), 2)
            )

            future_predictions.append({
                "product_id": product,
                "forecast_month": forecast_months[step].strftime("%Y-%m"),
                "predicted_purchases": pred,
            })

            lag_values = [pred] + lag_values[:-1]

    return pd.DataFrame(future_predictions)

# =============================================================================
# 🚀 FORECAST ENDPOINT (PROTECTED)
# =============================================================================
@app.post("/forecast")
async def forecast_endpoint(
    file: UploadFile = File(...),
    user=Depends(verify_user)   # 🔐 protected
):
    user_id = user["uid"]

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    contents = await file.read()

    df = validate_and_load(contents)
    df = engineer_features(df)

    predictions_df = forecast(df)

    if predictions_df.empty:
        raise HTTPException(422, "Not enough data per product")

    stream = io.StringIO()
    predictions_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=forecast_{user_id}.csv"
        },
    )

# =============================================================================
# 📊 METRICS ENDPOINT (PROTECTED)
# =============================================================================
@app.get("/metrics")
def get_metrics(user=Depends(verify_user)):
    return {
        "total_experiments": len(TRAINING_RESULTS),
        "results": TRAINING_RESULTS
    }