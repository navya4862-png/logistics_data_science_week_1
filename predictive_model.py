import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

DATA_FILE = "DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATA_FILE, encoding="latin1")

# Standardize names
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
)

# Verify exact source column names before running the model.
# The following names are common in the DataCo dataset.
target = "late_delivery_risk"

candidate_features = [
    "days_for_shipping_scheduled",
    "sales_per_customer",
    "benefit_per_order",
    "order_item_quantity",
    "order_item_discount_rate",
]

available_features = [c for c in candidate_features if c in df.columns]

if target not in df.columns:
    raise KeyError(
        f"Target '{target}' was not found. Inspect df.columns and update the script."
    )

if not available_features:
    raise ValueError("No candidate predictor columns were found.")

X = df[available_features].copy()
y = df[target].astype(int)

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

transformers = [
    ("num", numeric_pipe, numeric_features)
]

if categorical_features:
    transformers.append(("cat", categorical_pipe, categorical_features))

preprocess = ColumnTransformer(transformers)

model = Pipeline([
    ("preprocess", preprocess),
    ("classifier", LogisticRegression(max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

model.fit(X_train, y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
