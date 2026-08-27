import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_FILE = "DataCoSupplyChainDataset.csv"

df = pd.read_csv(DATA_FILE, encoding="latin1")

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing-value percentage:")
print((df.isna().mean() * 100).sort_values(ascending=False).head(20))

print("\nDuplicate rows:", df.duplicated().sum())

# Standardize column names
df.columns = (
    df.columns.str.strip()
              .str.lower()
              .str.replace(" ", "_")
)

print("\nStandardized columns:")
print(df.columns.tolist())

# Basic numerical summary
print("\nNumerical summary:")
print(df.describe(include="all").transpose().head(20))
