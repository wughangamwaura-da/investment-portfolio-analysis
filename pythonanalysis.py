import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv("investment_data.csv", encoding="latin1")

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
    .str.replace(' ', '_')
)

print("\nFIRST 5 ROWS")
print(df.head())

print("\nDATASET SHAPE BEFORE CLEANING")
print(df.shape)

numeric_cols = ['amount', 'current_price', 'initial_price']

for col in numeric_cols:

    df[col] = df[col].astype(str)

    df[col] = (
        df[col]
        .str.replace(',', '', regex=False)
        .str.replace('%', '', regex=False)
        .str.replace(r'[^0-9.\-]', '', regex=True)
        .str.strip()
    )

    df[col] = pd.to_numeric(df[col], errors='coerce')

print("\nMISSING VALUES")
print(df.isna().sum())

df = df.dropna(subset=["amount", "current_price", "initial_price"])

print("\nDATASET SHAPE AFTER CLEANING")
print(df.shape)

print("\nCLEANED NUMERIC COLUMNS")
print(df[['amount', 'current_price', 'initial_price']])

df["return"] = (
    (df["current_price"] - df["initial_price"])
    / df["initial_price"]
)

total_invested = df["amount"].sum()

print("\nTOTAL INVESTED")
print(total_invested)

df["allocation"] = df["amount"] / total_invested

print("\nALLOCATION")
print(df[["asset", "allocation"]])

print("\nCATEGORY BREAKDOWN")
print(df.groupby("type")["amount"].sum())

risk = df["return"].std()

print("\nPORTFOLIO RISK")
print(risk)

avg_return = df["return"].mean()

print("\nAVERAGE RETURN")
print(avg_return)

sharpe = avg_return / risk

print("\nSHARPE RATIO")
print(sharpe)

portfolio_return = (
    df["allocation"] * df["return"]
).sum()

print("\nPORTFOLIO RETURN")
print(portfolio_return)

plt.figure(figsize=(8, 5))

plt.scatter(df["return"], df["amount"])

plt.xlabel("Return")
plt.ylabel("Investment Size")
plt.title("Risk vs Exposure")

plt.grid(True)

plt.show()

df["cumulative_return"] = (
    1 + df["return"]
).cumprod()

plt.figure(figsize=(8, 5))

plt.plot(df["cumulative_return"])

plt.xlabel("Investment Period")
plt.ylabel("Cumulative Growth")
plt.title("Portfolio Growth Over Time")

plt.grid(True)

plt.show()

best_asset = df.loc[df["return"].idxmax()]

worst_asset = df.loc[df["return"].idxmin()]

print("\nBEST PERFORMING ASSET")
print(best_asset[["asset", "return"]])

print("\nWORST PERFORMING ASSET")
print(worst_asset[["asset", "return"]])

df.to_csv("analysis_results.csv", index=False)

print("\nANALYSIS COMPLETE")
print("Results saved to analysis_results.csv")