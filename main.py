import io
import asyncio
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.security import HTTPBearer
from fastapi.openapi.docs import get_swagger_ui_html
from pydantic import BaseModel

# 🔐 Firebase
import firebase_admin
from firebase_admin import credentials, auth

# 🏋️ Training function
from Smart_Za3bola import train_on_df




#import firebase_admin
#from firebase_admin import credentials

#if not firebase_admin._apps:
    #cred = credentials.Certificate("/etc/secrets/firebase")
    #firebase_admin.initialize_app(cred)
# =============================================================================
# 🔥  INIT FIREBASE
# =============================================================================

# Load Firebase key from environment variable
firebase_key_json = os.getenv("firebase")
if not firebase_key_json:
    raise ValueError("firebase environment variable not set")

import json
cred = credentials.Certificate(json.loads(firebase_key_json))
firebase_admin.initialize_app(cred, {
    'storageBucket': os.getenv('FIREBASE_STORAGE_BUCKET', 'sales-forecasting-75f26.appspot.com')
})

security = HTTPBearer()

# =============================================================================
# 📧  GMAIL CONFIG
# =============================================================================
GMAIL_SENDER   = os.getenv("GMAIL_SENDER")
GMAIL_APP_PASS = os.getenv("GMAIL_APP_PASS")

def send_forecast_email(to_email: str, forecast_csv: str, months: int, metrics: dict):
    msg = MIMEMultipart()
    msg["From"]    = GMAIL_SENDER
    msg["To"]      = to_email
    msg["Subject"] = f"📊 Your {months}-Month Sales Forecast is Ready"

    body = f"""
<html><body style="font-family:Arial,sans-serif;background:#03040a;color:#f1f5f9;padding:32px;">
  <div style="max-width:560px;margin:0 auto;background:#0b0f1e;border:1px solid rgba(124,58,255,.25);border-radius:16px;padding:32px;">
    <h2 style="color:#7c3aff;margin-bottom:4px;">Sales Forecast Ready 🚀</h2>
    <p style="color:#64748b;font-size:13px;margin-bottom:24px;">Your {months}-month forecast has been generated successfully.</p>
    <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:24px;">
      <tr style="border-bottom:1px solid rgba(255,255,255,.06);">
        <td style="padding:10px 0;color:#64748b;">Model</td>
        <td style="padding:10px 0;color:#f1f5f9;text-align:right;"><strong>{metrics.get('model_name','—')}</strong></td>
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,.06);">
        <td style="padding:10px 0;color:#64748b;">Val RMSE</td>
        <td style="padding:10px 0;color:#34d399;text-align:right;"><strong>{round(metrics.get('val_rmse',0),4)}</strong></td>
      </tr>
      <tr style="border-bottom:1px solid rgba(255,255,255,.06);">
        <td style="padding:10px 0;color:#64748b;">Baseline RMSE</td>
        <td style="padding:10px 0;color:#f1f5f9;text-align:right;">{round(metrics.get('baseline_rmse',0),4)}</td>
      </tr>
      <tr>
        <td style="padding:10px 0;color:#64748b;">Forecast Horizon</td>
        <td style="padding:10px 0;color:#06b6d4;text-align:right;"><strong>{months} months</strong></td>
      </tr>
    </table>
    <p style="color:#64748b;font-size:12px;">The full forecast is attached as a CSV file.</p>
    <p style="color:#64748b;font-size:11px;margin-top:24px;border-top:1px solid rgba(255,255,255,.06);padding-top:16px;">AI Demand Forecast API · Sent automatically after forecast generation</p>
  </div>
</body></html>
"""
    msg.attach(MIMEText(body, "html"))

    attachment = MIMEBase("application", "octet-stream")
    attachment.set_payload(forecast_csv.encode("utf-8"))
    encoders.encode_base64(attachment)
    attachment.add_header("Content-Disposition", f"attachment; filename=forecast_{months}mo.csv")
    msg.attach(attachment)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(GMAIL_SENDER, GMAIL_APP_PASS)
        server.sendmail(GMAIL_SENDER, to_email, msg.as_string())


# =============================================================================
# ☁️  CLOUD MODELS DIRECTORY 
# =============================================================================
import tempfile
from firebase_admin import storage

def upload_model(uid: str, bundle: dict):
    """رفع الـ model للـ Firebase Storage"""
    bucket = storage.bucket()
    blob = bucket.blob(f"user_models/{uid}.joblib")
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        joblib.dump(bundle, tmp_name)
        blob.upload_from_filename(tmp_name)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def download_model(uid: str):
    """تحميل الـ model من الـ Firebase Storage"""
    bucket = storage.bucket()
    blob = bucket.blob(f"user_models/{uid}.joblib")
    if not blob.exists():
        return None
    
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_name = tmp.name
    try:
        blob.download_to_filename(tmp_name)
        bundle = joblib.load(tmp_name)
        return bundle
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)

def delete_model_cloud(uid: str) -> bool:
    """حذف الـ model من الـ Firebase Storage"""
    bucket = storage.bucket()
    blob = bucket.blob(f"user_models/{uid}.joblib")
    if blob.exists():
        blob.delete()
        return True
    return False


# =============================================================================
# 🚀  FastAPI App
# =============================================================================
app = FastAPI(
    title="AI Demand Forecast API 🚀",
    description="Secure SaaS Forecasting API — train once, forecast anytime",
    version="5.0.0",
    swagger_ui_parameters={"persistAuthorization": True},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 🔐  VERIFY USER
# =============================================================================
def verify_user(token=Depends(security)):
    """
    يetحقق من الـ Firebase ID token.
    - check_revoked=False: أسرع وأكثر موثوقية (Firebase بيتحقق من الـ signature والـ expiry تلقائي)
    - لو التوكين انتهى (<1 hour) Firebase بيرفضه بـ ExpiredIdTokenError
    """
    try:
        decoded_token = auth.verify_id_token(token.credentials, check_revoked=False)
        return decoded_token
    except auth.ExpiredIdTokenError:
        raise HTTPException(
            status_code=401,
            detail="Token expired — please refresh the page or login again"
        )
    except auth.InvalidIdTokenError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {e}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {e}"
        )


# =============================================================================
# 📄  SWAGGER DOCS — auto-authorize from localStorage token
# =============================================================================
@app.get("/docs", include_in_schema=False)
async def custom_docs():
    html = get_swagger_ui_html(
        openapi_url="/openapi.json",
        title="My API"
    ).body.decode("utf-8")

    wrap_script = """
<script src="https://www.gstatic.com/firebasejs/8.10.0/firebase-app.js"></script>
<script src="https://www.gstatic.com/firebasejs/8.10.0/firebase-auth.js"></script>
<script>
(function(){
    // ── Init Firebase ──────────────────────────────────────────────────────
    var firebaseConfig = {
        apiKey: "AIzaSyBTi0hycT_nCgThOoLDLDfXhCuLWeKcPMU",
        authDomain: "sales-forecasting-75f26.firebaseapp.com"
    };
    if (!firebase.apps.length) firebase.initializeApp(firebaseConfig);

    // currentUser: مرجع للـ Firebase user الحالي عشان نجيب منه token fresh في أي وقت
    var currentUser = null;

    // getToken: بيجيب token fresh من Firebase مباشرة (مش من localStorage)
    // forceRefresh=false: يستخدم الـ cache لو التوكن لسه صالح (< 1 hour)
    function getToken(forceRefresh) {
        if (!currentUser) return Promise.resolve(null);
        return currentUser.getIdToken(forceRefresh || false);
    }

    function applyTokenToSwagger(token) {
        if (!token || !window.ui) return;
        try {
            window.ui.authActions.authorize({
                HTTPBearer: {
                    name: "HTTPBearer", value: token,
                    schema: {type: "http", scheme: "bearer"}
                }
            });
        } catch(e) { console.warn("Swagger authorize error:", e); }
    }

    // ── Auth state listener ─────────────────────────────────────────────────
    firebase.auth().onAuthStateChanged(function(user) {
        if (!user) {
            currentUser = null;
            console.warn("⚠️ No Firebase user — please login first.");
            return;
        }
        currentUser = user;
        console.log("🔥 Firebase user:", user.email, "| UID:", user.uid);

        // جيب token fresh وحدّث الـ Swagger
        getToken(true).then(function(token) {
            applyTokenToSwagger(token);
            console.log("✅ Swagger authorized for:", user.email);
        });

        // جدد التوكن كل 50 دقيقة (قبل انتهاء الـ 60 دقيقة)
        setInterval(function() {
            getToken(true).then(function(token) {
                applyTokenToSwagger(token);
                console.log("🔄 Token proactively refreshed for:", user.email);
            });
        }, 50 * 60 * 1000);
    });

    // ── Swagger Bundle wrapper ──────────────────────────────────────────────
    var waitForBundle = setInterval(function(){
        if (typeof SwaggerUIBundle === "undefined") return;
        clearInterval(waitForBundle);

        var _Orig = SwaggerUIBundle;
        window.SwaggerUIBundle = function(cfg) {
            var _ri = cfg.requestInterceptor;

            // ⬇ كل request: اجيب token fresh من Firebase (مش من localStorage)
            // ده بيضمن إن كل request بيتبعت بتوكن صالح
            cfg.requestInterceptor = function(req) {
                // بنرجع promise — Swagger بيدعم async interceptors
                return getToken(false).then(function(token) {
                    if (token) {
                        req.headers["Authorization"] = "Bearer " + token;
                    }
                    return _ri ? _ri(req) : req;
                });
            };

            var _oc = cfg.onComplete;
            cfg.onComplete = function() {
                // لو التوكن جاهز authorize فورًا
                getToken(false).then(function(token) {
                    applyTokenToSwagger(token);
                    console.log("✅ Swagger auto-authorized on load");
                });
                if (_oc) _oc();
            };

            var instance = _Orig(cfg);
            window.ui = instance;
            return instance;
        };
        Object.keys(_Orig).forEach(function(k) {
            try { window.SwaggerUIBundle[k] = _Orig[k]; } catch(e) {}
        });
    }, 30);
})();
</script>"""

    last_script_pos = html.rfind("<script>")
    html = html[:last_script_pos] + wrap_script + "\n" + html[last_script_pos:]
    return HTMLResponse(html)


# =============================================================================
# CONSTANTS
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


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
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


FORECAST_HORIZON_OPTIONS = [3, 6, 9, 12]

def run_forecast(df: pd.DataFrame, bundle: dict, months: int = 3) -> pd.DataFrame:
    if months not in FORECAST_HORIZON_OPTIONS:
        raise ValueError(f"months must be one of {FORECAST_HORIZON_OPTIONS}")

    model     = bundle["model"]
    BEST_LAGS = bundle["lags"]
    BEST_ROLL = bundle["roll"]

    last_month      = df["month"].max()
    forecast_months = [
        last_month + pd.DateOffset(months=i)
        for i in range(1, months + 1)
    ]

    future_predictions = []

    for product in df["product_id"].unique():
        product_df = df[df["product_id"] == product]
        if len(product_df) < BEST_LAGS:
            continue

        hist       = product_df.iloc[-BEST_LAGS:]
        lag_values = hist["number_of_product_purchases"].values.tolist()
        conv_rate  = hist["conversion_rate"].iloc[-1]
        drop_rate  = hist["cart_drop_rate"].iloc[-1]

        for step in range(months):
            row = {f"lag_{i+1}": lag_values[i] for i in range(BEST_LAGS)}
            row["product_id"] = product

            valid_lags = [v for v in lag_values[:BEST_ROLL] if not np.isnan(v)]
            row[f"rolling_mean_{BEST_ROLL}"] = np.mean(valid_lags) if valid_lags else np.nan
            row[f"rolling_std_{BEST_ROLL}"]  = np.std(valid_lags) if len(valid_lags) > 1 else np.nan
            row["conversion_rate"] = conv_rate
            row["cart_drop_rate"]  = drop_rate

            pred = max(0.0, round(float(model.predict(pd.DataFrame([row]))[0]), 2))

            future_predictions.append({
                "product_id":          product,
                "forecast_month":      forecast_months[step].strftime("%Y-%m"),
                "predicted_purchases": pred,
            })

            lag_values = [pred] + lag_values[:-1]

    return pd.DataFrame(future_predictions)


# =============================================================================
# 🏋️  TRAIN ENDPOINT — يعمل train مرة وبيحفظ الـ model للـ user
# =============================================================================
@app.post(
    "/train",
    summary="Upload CSV → train model → save it for your account",
    response_description="Training metrics for the saved model",
)
async def train_endpoint(
    file: UploadFile = File(..., description="CSV file with historical sales data"),
    user=Depends(verify_user),
):
    """
    **Flow:**
    1. استقبال الـ CSV وتحقق من الأعمدة
    2. Feature engineering (conversion_rate, cart_drop_rate)
    3. **Train** — يشغّل الـ grid search ويحفظ أحسن model
    4. يحفظ الـ model على disk باسم الـ user UID
    5. يرجّع الـ metrics
    """

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    contents = await file.read()

    # ── 1. Load & validate ────────────────────────────────────────────────
    df = validate_and_load(contents)

    # ── 2. Feature engineering ────────────────────────────────────────────
    df = engineer_features(df)

    # ── 3. Train — run in thread pool so we don't block the event loop ──────
    # train_on_df is a slow synchronous function (grid search).
    # Running it directly in async would freeze the entire server.
    try:
        bundle = await asyncio.to_thread(train_on_df, df)
    except Exception as exc:
        raise HTTPException(422, f"Training failed: {exc}")

    # ── 4. Save model to cloud for this user ──────────────────────────────
    # stamp owner info inside the bundle so we can verify isolation later
    bundle["owner_uid"]   = user["uid"]
    bundle["owner_email"] = user.get("email", "unknown")
    upload_model(user["uid"], bundle)

    # ── 5. Return metrics ─────────────────────────────────────────────────
    metrics = bundle["metrics"]
    return JSONResponse({
        "message":       "✅ Model trained and saved successfully",
        "uid":           user["uid"],
        "owner_email":   user.get("email", "unknown"),
        "model_name":    metrics["model_name"],
        "train_rmse":    round(metrics["train_rmse"],    4),
        "val_rmse":      round(metrics["val_rmse"],      4),
        "baseline_rmse": round(metrics["baseline_rmse"], 4),
        "best_lags":     bundle["lags"],
        "best_roll":     bundle["roll"],
    })


# =============================================================================
# 🚀  FORECAST ENDPOINT — يلود الـ model المحفوظ ويعمل predict مباشرة
# =============================================================================
@app.post(
    "/forecast",
    summary="Upload CSV → load your saved model → get forecast",
    response_description="CSV file with predicted purchases per product",
)
async def forecast_endpoint(
    file: UploadFile = File(..., description="CSV file with historical sales data"),
    months: int = Query(3, description="Forecast horizon in months", enum=FORECAST_HORIZON_OPTIONS),
    user=Depends(verify_user),
):
    """
    **Flow:**
    1. التحقق إن اليوزر عمل /train قبل كده
    2. تحميل الـ model المحفوظ من disk
    3. استقبال الـ CSV الجديد وعمل Feature engineering عليه
    4. **Forecast** مباشرة بدون أي retrain
    5. يرجّع CSV بالـ predictions

    > ⚠️ لازم تعمل `/train` الأول قبل ما تستخدم هذا الـ endpoint
    """

    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files allowed")

    # ── 1. Load user's saved model ────────────────────────────────────────
    bundle = download_model(user["uid"])
    if not bundle:
        raise HTTPException(
            404,
            "No trained model found for your account. "
            "Please call POST /train first with your historical data."
        )

    contents = await file.read()

    # ── 2. Load & validate ────────────────────────────────────────────────
    df = validate_and_load(contents)

    # ── 3. Feature engineering ────────────────────────────────────────────
    df = engineer_features(df)

    # ── 4. Forecast ───────────────────────────────────────────────────────
    preds_df = run_forecast(df, bundle, months=months)

    if preds_df.empty:
        raise HTTPException(422, "Not enough data per product to forecast")

    # ── 5. Return CSV ─────────────────────────────────────────────────────
    stream = io.StringIO()
    preds_df.to_csv(stream, index=False)
    stream.seek(0)

    metrics = bundle["metrics"]

    # ── 6. Send email (non-blocking) ─────────────────────────────────────────
    user_email = user.get("email")
    if user_email:
        try:
            await asyncio.to_thread(
                send_forecast_email,
                user_email,
                stream.getvalue(),
                months,
                metrics,
            )
        except Exception as mail_exc:
            print(f"⚠️ Email send failed (non-fatal): {mail_exc}")

    stream.seek(0)
    return StreamingResponse(
        iter([stream.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=forecast_{user['uid']}.csv",
            "X-Model-Name":    metrics["model_name"],
            "X-Train-RMSE":    str(round(metrics["train_rmse"],    4)),
            "X-Val-RMSE":      str(round(metrics["val_rmse"],      4)),
            "X-Baseline-RMSE": str(round(metrics["baseline_rmse"], 4)),
            "X-Best-Lags":     str(bundle["lags"]),
            "X-Best-Roll":     str(bundle["roll"]),
            "X-Forecast-Months": str(months),
        },
    )


# =============================================================================
# 🗑️  DELETE MODEL ENDPOINT — يحذف الـ model المحفوظ للـ user
# =============================================================================
@app.delete(
    "/model",
    summary="Delete your saved model",
)
async def delete_model(user=Depends(verify_user)):
    """
    يحذف الـ model المحفوظ للـ user — بعد كده لازم يعمل /train تاني.
    مفيد لو اليوزر عايز يعيد الـ training على داتا جديدة من الأساس.
    """
    deleted = delete_model_cloud(user["uid"])
    if not deleted:
        raise HTTPException(404, "No model found to delete.")
    return JSONResponse({"message": "✅ Model deleted. You can now retrain with new data."})


# =============================================================================
# 📊  METRICS ENDPOINT (PROTECTED)
# =============================================================================
@app.get("/metrics", summary="Get your saved model's metrics")
def get_metrics(user=Depends(verify_user)):
    """
    يرجّع الـ metrics الخاصة بالـ model المحفوظ للـ user.
    """
    bundle = download_model(user["uid"])
    if not bundle:
        raise HTTPException(
            404,
            "No trained model found. Please call POST /train first."
        )

    metrics = bundle["metrics"]

    return JSONResponse({
        "model_name":    metrics["model_name"],
        "owner_email":   bundle.get("owner_email", "unknown"),
        "train_rmse":    round(metrics["train_rmse"],    4),
        "val_rmse":      round(metrics["val_rmse"],      4),
        "baseline_rmse": round(metrics["baseline_rmse"], 4),
        "best_lags":     bundle["lags"],
        "best_roll":     bundle["roll"],
    })

