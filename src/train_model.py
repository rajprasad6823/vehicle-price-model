# =============================
# IMPORTS
# =============================
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

# =============================
# CONFIG
# =============================
TRAIN_PATH = r"C:\Users\Rajprasad\Desktop\Pickles\train_pickles.csv"
TEST_PATH  = r"C:\Users\Rajprasad\Desktop\Pickles\test_pickles.csv"

TARGET = "Sold_Amount"
NEWPRICE_COL = "NewPrice"  # column containing reference price
TEXT_COLUMNS = [
    "Make",	"Model",	"MakeCode"	,"FamilyCode",

   
]

IGNORE_COLS = [
    TARGET, "VIN", "SequenceNum", "ModelCode",
    "Sold_Date", "Compliance_Date","Description", "BadgeDescription",
     "BodyStyleDescription"
]

BAD_TOKENS = {"t", "Y", "N", "?"}
MAX_SHAP = 2000

# =============================
# UTILS
# =============================
def clean_numeric_tokens(df, numeric_cols):
    df = df.copy()
    for col in numeric_cols:
        df[col] = df[col].replace(list(BAD_TOKENS), np.nan)
    return df

def mape_safe(y_true, y_pred, eps=1e-6):
    y_true = np.asarray(y_true)
    mask = y_true > eps
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_abs_shap(shap_values, feature_names, top_n=20):
    imp = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
    )

def get_feature_names(preprocessor):
    feature_names = []

    # numeric
    feature_names.extend(numeric_cols)

    # categorical (OHE)
    ohe = preprocessor.named_transformers_["cat"]
    cat_features = ohe.get_feature_names_out(cat_cols)
    feature_names.extend(cat_features.tolist())

    # embeddings after PCA
    # pca_components = preprocessor.named_transformers_["emb"].n_components_
    # feature_names.extend([f"emb_pca_{i}" for i in range(pca_components)])

    return feature_names

# =============================
# LOAD TRAIN
# =============================
train_df = pd.read_csv(TRAIN_PATH).dropna(subset=[TARGET, NEWPRICE_COL])
train_df[TARGET] = train_df[TARGET].astype(float)
train_df[NEWPRICE_COL] = train_df[NEWPRICE_COL].astype(float)

feature_cols = [c for c in train_df.columns if c not in IGNORE_COLS]
numeric_cols = train_df[feature_cols].select_dtypes(include=["int64","float64"]).columns.tolist()
cat_cols = train_df[feature_cols].select_dtypes(include=["object"]).columns.tolist()

train_df = clean_numeric_tokens(train_df, numeric_cols)

# =============================
# CREATE NEW TARGET
# =============================
train_df["Target_Ratio"] = train_df[TARGET] / train_df[NEWPRICE_COL]


# =============================
# PREPROCESSOR
# =============================
preprocessor = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    
    ]
)

X_train = train_df.drop(columns=[TARGET, "Target_Ratio"])
y_train = train_df["Target_Ratio"]

X_proc = preprocessor.fit_transform(X_train)
FEATURE_NAMES = get_feature_names(preprocessor)

# =============================
# SINGLE MODEL
# =============================
model = LGBMRegressor(
    objective="regression",
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=64,
    random_state=42
)

model.fit(X_proc, y_train)

# =============================
# LOAD TEST
# =============================
test_df = pd.read_csv(TEST_PATH).dropna(subset=[TARGET, NEWPRICE_COL])
test_df[TARGET] = test_df[TARGET].astype(float)
test_df[NEWPRICE_COL] = test_df[NEWPRICE_COL].astype(float)

test_df = clean_numeric_tokens(test_df, numeric_cols)

# TEXT EMBEDDINGS
text = test_df[TEXT_COLUMNS].fillna("").astype(str).agg(" ".join, axis=1)


X_test = test_df.drop(columns=[TARGET])
X_test_proc = preprocessor.transform(X_test)

# =============================
# PREDICTIONS
# =============================
pred_ratio = model.predict(X_test_proc)
pred_sold_amount = pred_ratio * test_df[NEWPRICE_COL].values
test_df["Pred_SoldAmount"] = pred_sold_amount

# =============================
# METRICS
# =============================
print("OVERALL PERFORMANCE")
print(f"MAE   : {mean_absolute_error(test_df[TARGET], pred_sold_amount):,.0f}")
print(f"RMSE  : {rmse(test_df[TARGET], pred_sold_amount):,.0f}")
print(f"MAPE  : {mape_safe(test_df[TARGET], pred_sold_amount):.2f}%")

# =============================
# OPTIONAL: SHAP
# =============================
X_shap = X_proc[:MAX_SHAP].toarray() if hasattr(X_proc, "toarray") else X_proc[:MAX_SHAP]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_shap)

shap.summary_plot(
    shap_values,
    X_shap,
    feature_names=FEATURE_NAMES,
    show=False
)
plt.title("SHAP – Single Model Predicting SoldAmount/NewPrice")
plt.tight_layout()
plt.show()

print("\nTOP FEATURES")
print(mean_abs_shap(shap_values, FEATURE_NAMES))

def mape_by_price_band(
    df,
    y_true_col,
    y_pred_col,
    bands
):
    """
    df: DataFrame containing true & predicted values
    bands: dict {label: (low, high)}
    """

    results = []

    for label, (low, high) in bands.items():
        mask = (df[y_true_col] >= low) & (df[y_true_col] < high)

        if mask.sum() == 0:
            results.append({
                "Price_Band": label,
                "Count": 0,
                "MAPE (%)": np.nan
            })
            continue

        mape = mape_safe(
            df.loc[mask, y_true_col],
            df.loc[mask, y_pred_col]
        )

        results.append({
            "Price_Band": label,
            "Count": mask.sum(),
            "MAPE (%)": round(mape, 2)
        })

    return pd.DataFrame(results)

PRICE_BANDS = {
    "LOW (0–10k)":    (0, 10_000),
    "MID (10k–30k)":  (10_000, 30_000),
    "HIGH (30k+)":    (30_000, np.inf)
}

print("\nMAPE BY PRICE BAND")
mape_band_df = mape_by_price_band(
    df=test_df,
    y_true_col=TARGET,
    y_pred_col="Pred_SoldAmount",
    bands=PRICE_BANDS
)

print(mape_band_df)
