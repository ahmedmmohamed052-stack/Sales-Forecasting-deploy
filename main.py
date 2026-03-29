import io
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.security import HTTPBearer
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel
# 🔐 Firebase
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin import credentials
import firebase_admin
from firebase_admin import credentials
# 🗄️ MySQL
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session


# =============================================================================
# 🔥  INIT FIREBASE
# =============================================================================
cred = credentials.Certificate(r"c:\Users\Ahmed\Desktop\firebase_key.json")
firebase_admin.initialize_app(cred)


security = HTTPBearer()


# =============================================================================
# 🚀  FastAPI App
# =============================================================================
app = FastAPI(
    title="AI Demand Forecast API 🚀",
    description="Secure SaaS Forecasting API",
    version="3.0.0",
    swagger_ui_parameters={"persistAuthorization": True},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # ← في Production حطّ domain بتاعك
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 🔐  VERIFY USER (للـ forecast والـ metrics بس)
# =============================================================================
def verify_user(token=Depends(security)):
    try:
        decoded_token = auth.verify_id_token(token.credentials)
        return decoded_token
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# =============================================================================
# 📄  SWAGGER DOCS
# =============================================================================
@app.get("/docs", include_in_schema=False)
async def custom_docs():
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="My API"
    ).body.decode("utf-8")

    # ══════════════════════════════════════════════════════
    # الحل النهائي: نعمل wrap للـ SwaggerUIBundle قبل الـ init
    # بنحط السكريبت قبل "<script>" اللي فيه "const ui = SwaggerUIBundle"
    # ══════════════════════════════════════════════════════

    wrap_script = """<script>
(function(){
    var t = localStorage.getItem("token");
    if(!t) return;

    // بنستنى SwaggerUIBundle يتحمّل من الـ CDN
    var waitForBundle = setInterval(function(){
        if(typeof SwaggerUIBundle === "undefined") return;
        clearInterval(waitForBundle);

        // نعمل wrap
        var _Orig = SwaggerUIBundle;
        window.SwaggerUIBundle = function(cfg){

            // ① requestInterceptor — مضمون 100%
            // كل request من Swagger هيكون فيه Authorization header
            var _ri = cfg.requestInterceptor;
            cfg.requestInterceptor = function(req){
                req.headers["Authorization"] = "Bearer " + t;
                return _ri ? _ri(req) : req;
            };

            // ② onComplete — يعمل authorize في الـ UI لما يخلص
            var _oc = cfg.onComplete;
            cfg.onComplete = function(){
                try {
                    window.ui.authActions.authorize({
                        HTTPBearer:{
                            name:"HTTPBearer",
                            value:t,
                            schema:{type:"http",scheme:"bearer"}
                        }
                    });
                    console.log("✅ Swagger auto-authorized");
                } catch(e){ console.warn("authorize err:", e); }
                if(_oc) _oc();
            };

            var instance = _Orig(cfg);
            window.ui = instance;
            return instance;
        };
        // نحافظ على كل properties الأصلية (presets, plugins, etc.)
        Object.keys(_Orig).forEach(function(k){
            try{ window.SwaggerUIBundle[k] = _Orig[k]; }catch(e){}
        });

    }, 30);  // بيفحص كل 30ms لحد ما SwaggerUIBundle يتعرّف
})();
</script>"""

    # نحط الـ wrap script مباشرة قبل آخر <script> في الـ HTML
    # (اللي هو الـ init script اللي فيه SwaggerUIBundle call)
    last_script_pos = html.rfind("<script>")
    html = html[:last_script_pos] + wrap_script + "\n" + html[last_script_pos:]
    return HTMLResponse(html)


# =============================================================================
# ✅  REGISTER USER — مخفي تماماً عن Swagger
# =============================================================================



# =============================================================================
# LOAD MODEL
# =============================================================================
try:
    bundle           = joblib.load("model_bundle.pkl")
    best_model       = bundle["model"]
    BEST_LAGS        = bundle["lags"]
    BEST_ROLL        = bundle["roll"]
    TRAINING_RESULTS = bundle["results"]
except Exception:
    raise Exception("❌ model_bundle.pkl not found. Run train.py first.")


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
    return df.drop(columns=[
        "number_of_times_add_followed_by_purchase",
        "number_of_times_add_followed_by_no_purchase",
    ])


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

        hist       = product_df.iloc[-BEST_LAGS:]
        lag_values = hist["number_of_product_purchases"].values.tolist()
        conv_rate  = hist["conversion_rate"].iloc[-1]
        drop_rate  = hist["cart_drop_rate"].iloc[-1]

        for step in range(2):
            row = {f"lag_{i+1}": lag_values[i] for i in range(BEST_LAGS)}
            row["product_id"] = product

            valid_lags = [v for v in lag_values[:BEST_ROLL] if not np.isnan(v)]
            row[f"rolling_mean_{BEST_ROLL}"] = np.mean(valid_lags) if valid_lags else np.nan
            row[f"rolling_std_{BEST_ROLL}"]  = np.std(valid_lags) if len(valid_lags) > 1 else np.nan
            row["conversion_rate"] = conv_rate
            row["cart_drop_rate"]  = drop_rate

            pred = max(0.0, round(float(best_model.predict(pd.DataFrame([row]))[0]), 2))

            future_predictions.append({
                "product_id"         : product,
                "forecast_month"     : forecast_months[step].strftime("%Y-%m"),
                "predicted_purchases": pred,
            })

            lag_values = [pred] + lag_values[:-1]

    return pd.DataFrame(future_predictions)


# =============================================================================
# 🚀  FORECAST ENDPOINT (PROTECTED — بيظهر في Swagger)
# =============================================================================
@app.post("/forecast")
async def forecast_endpoint(
    file: UploadFile = File(...),
    user=Depends(verify_user),
):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    contents = await file.read()
    df       = validate_and_load(contents)
    df       = engineer_features(df)
    preds_df = forecast(df)

    if preds_df.empty:
        raise HTTPException(422, "Not enough data per product")

    stream = io.StringIO()
    preds_df.to_csv(stream, index=False)
    stream.seek(0)

    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=forecast_{user['uid']}.csv"},
    )


# =============================================================================
# 📊  METRICS ENDPOINT (PROTECTED — بيظهر في Swagger)
# =============================================================================
@app.get("/metrics")
def get_metrics(user=Depends(verify_user)):
    return {
        "total_experiments": len(TRAINING_RESULTS),
        "results"          : TRAINING_RESULTS,
    }