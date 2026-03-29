import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import joblib


df = pd.read_csv(
    r'c:\Users\Ahmed\Desktop\E-Commerce Services\Inventory and Offers and discounts decisions\ecommerce_synthetic_seasonal_hard_dataset.csv'
)

df = df.drop_duplicates()
df['month'] = pd.to_datetime(df['month'])
df = df.sort_values(["product_id", "month"]).reset_index(drop=True)

df['conversion_rate'] = (
    df['number_of_times_add_followed_by_purchase'] /
    df['number_of_times_added_to_cart'].replace(0, np.nan)
)

df['cart_drop_rate'] = (
    df['number_of_times_add_followed_by_no_purchase'] /
    df['number_of_times_added_to_cart'].replace(0, np.nan)
)

df = df.drop(columns=[
    'number_of_times_add_followed_by_purchase',
    'number_of_times_add_followed_by_no_purchase'
])


LAG_OPTIONS = [8, 10, 12]
ROLLING_OPTIONS = [3, 6, 9]

MODELS = {
    "ridge": {
        "model": Ridge(),
        "params": {
            "model__alpha": [0.1, 0.5, 1.0, 1.5]
        }
    },
    "xgb": {
        "model": XGBRegressor(
            random_state=42,
            objective="reg:squarederror"
        ),
        "params": {
            'model__n_estimators': [20, 35, 50, 75, 100, 120, 135, 150],
            'model__max_depth': [2, 4, 6, 8, 10],
            'model__learning_rate': [0.03, 0.05, 0.1, 0.2],
            'model__subsample': [0.6, 0.8, 1.0],
            'model__colsample_bytree': [0.6, 0.8, 1.0]
        }
    }
}


numerical_pipeline = Pipeline(steps=[
    ('imputer', IterativeImputer(random_state=0)),
    ('scaler', StandardScaler())
])

categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

tscv = TimeSeriesSplit(n_splits=5)

results = []


for LAGS in LAG_OPTIONS:
    for ROLL in ROLLING_OPTIONS:

        temp_df = df.copy()

        for lag in range(1, LAGS + 1):
            temp_df[f"lag_{lag}"] = (
                temp_df.groupby("product_id")
                ["number_of_product_purchases"]
                .shift(lag)
            )

        temp_df[f'rolling_mean_{ROLL}'] = (
            temp_df.groupby('product_id')
            ['number_of_product_purchases']
            .rolling(ROLL).mean()
            .reset_index(0, drop=True)
        )

        temp_df[f'rolling_std_{ROLL}'] = (
            temp_df.groupby('product_id')
            ['number_of_product_purchases']
            .rolling(ROLL).std()
            .reset_index(0, drop=True)
        )

        temp_df = temp_df.dropna().reset_index(drop=True)

        x = temp_df.drop(
            columns=['number_of_product_purchases', 'month']
        )
        y = temp_df['number_of_product_purchases']

        split = int(len(x) * 0.8)

        x_train = x.iloc[:split]
        x_val = x.iloc[split:]
        y_train = y.iloc[:split]
        y_val = y.iloc[split:]

        if len(y_val) <= 1:
            continue

        y_val_baseline = y_val.shift(1).iloc[1:]
        y_val_true = y_val.iloc[1:]

        baseline_rmse = np.sqrt(mean_squared_error(
            y_val_true, y_val_baseline
        ))

        num_features = (
            [f"lag_{i}" for i in range(1, LAGS + 1)]
            + [
                f"rolling_mean_{ROLL}",
                f"rolling_std_{ROLL}",
                "conversion_rate",
                "cart_drop_rate"
            ]
        )

        preprocessor = ColumnTransformer(transformers=[
            ('num', numerical_pipeline, num_features),
            ('cat', categorical_pipeline, ['product_id'])
        ])

        for model_name, model_info in MODELS.items():

            model_pipe = Pipeline([
                ("prep", preprocessor),
                ("model", model_info["model"])
            ])

            gridsearch = GridSearchCV(
                model_pipe,
                param_grid=model_info["params"],
                cv=tscv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            gridsearch.fit(x_train, y_train)

            best_model = gridsearch.best_estimator_

            train_rmse = np.sqrt(mean_squared_error(
                y_train,
                best_model.predict(x_train)
            ))

            val_rmse = np.sqrt(mean_squared_error(
                y_val,
                best_model.predict(x_val)
            ))

            results.append({
                "model": model_name,
                "lags": LAGS,
                "rolling": ROLL,
                "baseline_rmse": baseline_rmse,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
                "best_params": gridsearch.best_params_,
                "estimator": best_model
            })


results_df = pd.DataFrame(results)
results_df["composite_score"] = (
    results_df["val_rmse"]
    + 0.5 * abs(results_df["val_rmse"] - results_df["train_rmse"])
    + 0.2 * (results_df["val_rmse"] / results_df["baseline_rmse"])
)

best_row = results_df.sort_values("composite_score").iloc[0]

best_model = best_row["estimator"]
BEST_LAGS = best_row["lags"]
BEST_ROLL = best_row["rolling"]

print("BEST CONFIGURATION")
print(best_row[
    ["model", "lags", "rolling",
     "baseline_rmse", "train_rmse", "val_rmse"]
])

final_baseline_rmse = best_row["baseline_rmse"]
final_train_rmse = best_row["train_rmse"]
final_val_rmse = best_row["val_rmse"]

print("\nFINAL METRICS")
print("Baseline RMSE     :", final_baseline_rmse)
print("Final Train RMSE  :", final_train_rmse)
print("Final Val RMSE    :", final_val_rmse)


# =======================
# ⭐⭐ التعديل الأول (حفظ الأعمدة)
# =======================
FEATURE_COLUMNS = x_train.columns.tolist()


# =======================
# FUTURE FORECAST
# =======================
future_predictions = []

for product in df['product_id'].unique():

    product_df = df[df['product_id'] == product]

    if len(product_df) < BEST_LAGS:
        continue

    hist = product_df.iloc[-BEST_LAGS:]
    lag_values = hist['number_of_product_purchases'].values.tolist()

    conv_rate = hist['conversion_rate'].iloc[-1]
    drop_rate = hist['cart_drop_rate'].iloc[-1]

    for step in range(1, 3):

        row = {
            f'lag_{i+1}': lag_values[i]
            for i in range(BEST_LAGS)
        }

        row['product_id'] = product

        # =======================
        # ⭐⭐ التعديل الثاني (rolling آمن)
        # =======================
        valid_lags = [v for v in lag_values[:BEST_ROLL] if not np.isnan(v)]

        row[f'rolling_mean_{BEST_ROLL}'] = (
            np.mean(valid_lags) if len(valid_lags) > 0 else np.nan
        )

        row[f'rolling_std_{BEST_ROLL}'] = (
            np.std(valid_lags) if len(valid_lags) > 1 else np.nan
        )

        row['conversion_rate'] = conv_rate
        row['cart_drop_rate'] = drop_rate

        input_df = pd.DataFrame([row])

        pred = best_model.predict(input_df)[0]

        future_predictions.append({
            'product_id': product,
            'month_ahead': step,
            'prediction': pred
        })

        lag_values = [pred] + lag_values[:-1]


future_predictions_df = pd.DataFrame(future_predictions)

future_predictions_df.to_excel(
    'future_predictions.xlsx',
    index=False
)

joblib.dump({
    "model": best_model,
    "lags": BEST_LAGS,
    "roll": BEST_ROLL,
    "feature_columns": FEATURE_COLUMNS,
    "results": results_df.drop(columns=["estimator"]).to_dict(orient="records")
}, "model_bundle.pkl")
print("Model saved successfully")


dff = pd.read_excel('future_predictions.xlsx')
#print(future_predictions_df)


# مجموع المبيعات لكل منتج
product_sales = dff.groupby('product_id')['prediction'].sum().sort_values(ascending=False)

top3_products = product_sales.head(3)
#print(top3_products)
#plt.figure(figsize=(8,5))
#plt.bar(top3_products.index, top3_products.values, color='green')
#plt.title('Top 3 Products by Total Purchases')
#plt.xlabel('Product ID')
#plt.ylabel('Total Purchases')



bottom3_products = product_sales.tail(3)
#print(bottom3_products)
#plt.figure(figsize=(8,5))
#plt.bar(bottom3_products.index, bottom3_products.values, color='red')
#plt.title('Bottom 3 Products by Total Purchases')
#plt.xlabel('Product ID')
#plt.ylabel('Total Purchases')
#plt.show()