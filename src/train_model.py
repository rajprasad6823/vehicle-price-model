# =============================
# IMPORTS
# =============================
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

# =============================
# CONFIG
# =============================
TRAIN_PATH = r"data\DatiumTrain.rpt"
TEST_PATH  = r"data\DatiumTest.rpt"

TARGET = "Sold_Amount"
NEWPRICE_COL = "NewPrice"

IGNORE_COLS = [
    TARGET, NEWPRICE_COL,
    "VIN", "SequenceNum", "ModelCode",
    "Sold_Date", "Compliance_Date",
    "Description", "BadgeDescription",
    "BodyStyleDescription","AvgWholesale",	"AvgRetail",	"GoodWholesale",
    "GoodRetail",	"TradeMin",	"TradeMax",	"PrivateMax"

]

BAD_TOKENS = {"t", "Y", "N", "?"}
MAX_SHAP = 2000

# =============================
# UTILS
# =============================
# =============================
# SHAP AGGREGATION (ORIGINAL COLUMNS)
# =============================
def build_ohe_mapping(preprocessor, numeric_cols, categorical_cols):
    mapping = {}

    # numeric features
    for col in numeric_cols:
        mapping[col] = col

    ohe = preprocessor.named_transformers_["cat"]
    ohe_features = ohe.get_feature_names_out(categorical_cols)

    for f in ohe_features:
        # sklearn format: colname_value
        col = f.split("_", maxsplit=1)[0]
        mapping[f] = col

    return mapping


def aggregate_shap_by_original_feature(shap_values, feature_names, feature_map):
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": np.abs(shap_values).mean(axis=0)
    })

    shap_df["original_feature"] = shap_df["feature"].map(feature_map)

    agg = (
        shap_df
        .groupby("original_feature", as_index=False)["mean_abs_shap"]
        .sum()
        .sort_values("mean_abs_shap", ascending=False)
    )

    return agg
def infer_column_types(df, ignore_cols, numeric_threshold=0.8):
    numeric_cols = []
    categorical_cols = []

    for col in df.columns:
        if col in ignore_cols:
            continue

        # Explicit string/object columns → categorical
        if df[col].dtype == "object":
            categorical_cols.append(col)
            continue

        # Try numeric coercion
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().mean() > numeric_threshold:
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    return numeric_cols, categorical_cols


def clean_numeric_tokens(df, numeric_cols):
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .replace(list(BAD_TOKENS), np.nan)
                .pipe(pd.to_numeric, errors="coerce")
            )
    return df


def mape_safe(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    mask = y_true > eps
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def drop_high_cardinality(df, cat_cols, max_unique=500):
    return [c for c in cat_cols if df[c].nunique() <= max_unique]


def get_feature_names(preprocessor, numeric_cols, categorical_cols):
    names = list(numeric_cols)
    ohe = preprocessor.named_transformers_["cat"]
    names.extend(ohe.get_feature_names_out(categorical_cols))
    return names


# =============================
# LOAD & PREPARE TRAIN
# =============================
train_df = pd.read_csv(TRAIN_PATH, sep="\t", low_memory=False)
print("Train length original:",len(train_df.columns))
train_df = train_df.dropna(subset=[TARGET, NEWPRICE_COL])

train_df[TARGET] = train_df[TARGET].astype(float)
train_df[NEWPRICE_COL] = train_df[NEWPRICE_COL].astype(float)

# Target engineering
train_df["Target_Ratio"] = train_df[TARGET] / train_df[NEWPRICE_COL]

X_train = train_df.drop(columns=[TARGET, "Target_Ratio"])
X_train = X_train.drop(columns=IGNORE_COLS, errors="ignore")

numeric_cols, categorical_cols = infer_column_types(X_train, IGNORE_COLS)
print("Initial numeric:", len(numeric_cols))
print("Initial categorical:", len(categorical_cols))

categorical_cols = drop_high_cardinality(X_train, categorical_cols, max_unique=100)
print("Categorical cols after drop:", len(categorical_cols))

# Drop the actual columns from X_train
X_train = X_train[numeric_cols + categorical_cols]
print("Train length after dropping high-cardinality:", len(X_train.columns))

# Clean numerics & cast categoricals
X_train = clean_numeric_tokens(X_train, numeric_cols)
X_train[categorical_cols] = X_train[categorical_cols].astype(str)

y_train = train_df["Target_Ratio"]
OUTPUT_PATH = "train_predictions.csv"
X_train.to_csv(OUTPUT_PATH, index=False)
# =============================
# PREPROCESSOR
# =============================
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

X_proc = preprocessor.fit_transform(X_train)
FEATURE_NAMES = get_feature_names(preprocessor, numeric_cols, categorical_cols)

# =============================
# MODEL
# =============================
model = LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_proc, y_train)

# =============================
# LOAD & PREPARE TEST
# =============================
test_df = pd.read_csv(TEST_PATH, sep="\t", low_memory=False)
test_df = test_df.dropna(subset=[TARGET, NEWPRICE_COL])

test_df[TARGET] = test_df[TARGET].astype(float)
test_df[NEWPRICE_COL] = test_df[NEWPRICE_COL].astype(float)

X_test = test_df.drop(columns=[TARGET])
X_test = clean_numeric_tokens(X_test, numeric_cols)
X_test[categorical_cols] = X_test[categorical_cols].astype(str)

X_test = X_test.reindex(columns=X_train.columns, fill_value=np.nan)
X_test_proc = preprocessor.transform(X_test)

# =============================
# PREDICTIONS
# =============================
pred_ratio = model.predict(X_test_proc)
test_df["Pred_SoldAmount"] = pred_ratio * test_df[NEWPRICE_COL]

# =============================
# METRICS
# =============================
print("\nOVERALL PERFORMANCE")
print(f"MAE   : {mean_absolute_error(test_df[TARGET], test_df['Pred_SoldAmount']):,.0f}")
print(f"RMSE  : {rmse(test_df[TARGET], test_df['Pred_SoldAmount']):,.0f}")
print(f"MAPE  : {mape_safe(test_df[TARGET], test_df['Pred_SoldAmount']):.2f}%")

# =============================
# SHAP
# =============================
# =============================
# SHAP (AGGREGATED TO ORIGINAL COLUMNS)
# =============================
X_shap = (
    X_proc[:MAX_SHAP].toarray()
    if hasattr(X_proc, "toarray")
    else X_proc[:MAX_SHAP]
)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(
    X_shap,
    check_additivity=False
)

# Build mapping from OHE → original columns
feature_map = build_ohe_mapping(
    preprocessor,
    numeric_cols,
    categorical_cols
)

# Aggregate SHAP to original features
shap_agg = aggregate_shap_by_original_feature(
    shap_values,
    FEATURE_NAMES,
    feature_map
)

print("\nTOP FEATURES (ORIGINAL COLUMNS)")
print(shap_agg.head(20))

# Plot aggregated SHAP
top = shap_agg.head(15)

plt.figure(figsize=(8, 6))
plt.barh(top["original_feature"], top["mean_abs_shap"])
plt.gca().invert_yaxis()
plt.xlabel("Mean |SHAP|")
plt.title("SHAP Importance by Original Feature")
plt.tight_layout()
plt.show()

# Optional export
shap_agg.to_csv("shap_importance_original_columns.csv", index=False)

# =============================
# MAPE BY PRICE BAND
# =============================
def mape_by_price_band(df, y_true, y_pred, bands):
    rows = []
    for label, (low, high) in bands.items():
        mask = (df[y_true] >= low) & (df[y_true] < high)
        rows.append({
            "Price_Band": label,
            "Count": int(mask.sum()),
            "MAPE (%)": round(mape_safe(df.loc[mask, y_true], df.loc[mask, y_pred]), 2)
            if mask.any() else np.nan
        })
    return pd.DataFrame(rows)


PRICE_BANDS = {
    "LOW (0–10k)":   (0, 10_000),
    "MID (10k–30k)": (10_000, 30_000),
    "HIGH (30k+)":   (30_000, np.inf)
}

print("\nMAPE BY PRICE BAND")
print(
    mape_by_price_band(
        test_df,
        TARGET,
        "Pred_SoldAmount",
        PRICE_BANDS
    )
)
