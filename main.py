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
# 🗄️ MySQL
from sqlalchemy import create_engine, Column, String, DateTime, Boolean, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# =============================================================================
# ⚙️  DATABASE SETUP
# =============================================================================
DATABASE_URL = "mysql+pymysql://root:ahmed2628@localhost:3305/sales_forecast_db"

engine      = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


# =============================================================================
# 🗂️  جدول users
# =============================================================================
class UserDB(Base):
    __tablename__ = "users"

    uid            = Column(String(128), primary_key=True, index=True)
    email          = Column(String(255), unique=True, nullable=False)
    display_name   = Column(String(255), nullable=True)
    email_verified = Column(Boolean, default=False)
    created_at     = Column(DateTime, default=datetime.utcnow)
    last_login     = Column(DateTime, nullable=True)

Base.metadata.create_all(bind=engine)   # بينشئ الجدول أوتوماتيك لو مش موجود


# =============================================================================
# 🔁  DB Dependency
# =============================================================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


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
# ✅  REGISTER USER — مخفي تماماً عن Swagger
# =============================================================================
class RegisterRequest(BaseModel):
    uid          : str
    email        : str
    display_name : str | None = None

@app.post(
    "/internal/register-user",   # ← /internal/ عشان واضح إنه مش للعموم
    include_in_schema=False,     # ← مش هيظهر في Swagger خالص
    status_code=201,
)
def register_user(payload: RegisterRequest, db: Session = Depends(get_db)):
    """
    بيتنادى من الـ HTML تلقائياً بعد Sign Up من Firebase.
    اليوزر مش هيشوفه في Swagger.
    """
    # لو موجود قبل كده — مش هنضيف تاني
    existing = db.query(UserDB).filter(UserDB.uid == payload.uid).first()
    if existing:
        return {"status": "exists", "uid": payload.uid}

    new_user = UserDB(
        uid          = payload.uid,
        email        = payload.email,
        display_name = payload.display_name,
        created_at   = datetime.utcnow(),
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return {"status": "registered", "uid": new_user.uid}


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