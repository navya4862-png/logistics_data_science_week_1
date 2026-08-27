import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_FILE = "DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATA_FILE, encoding="latin1")

df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
)

candidate_features = [
    "sales_per_customer",
    "benefit_per_order",
    "days_for_shipping_real",
]

available = [c for c in candidate_features if c in df.columns]

if len(available) < 2:
    raise ValueError("Not enough clustering variables were found.")

X = df[available].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

labels = kmeans.fit_predict(X_scaled)

result = X.copy()
result["cluster"] = labels

print("Cluster profiles:")
print(result.groupby("cluster").mean(numeric_only=True))
