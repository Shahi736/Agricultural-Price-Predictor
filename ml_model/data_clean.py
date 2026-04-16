import pandas as pd
import os

print("Working directory:", os.getcwd())

file_path = "real_data.csv"

if not os.path.exists(file_path):
    print("ERROR: real_data.csv not found!")
    print("Please put real_data.csv in this folder:", os.getcwd())
    exit()

# Load data
df = pd.read_csv(file_path)
print("Columns found:", df.columns.tolist())
print("Total rows:", len(df))

# Clean column names (remove spaces)
df.columns = df.columns.str.strip()

# Rename to standard names - matches your exact CSV columns
df.rename(columns={
    "STATE":        "state",
    "Market Name":  "market",
    "Commodity":    "crop",
    "Modal_Price":  "price",
    "Price Date":   "date"
}, inplace=True)

# Remove missing values
df = df.dropna()

# Clean text - Title Case is very important (must match api.py)
df["crop"]   = df["crop"].str.strip().str.title()
df["state"]  = df["state"].str.strip().str.title()
df["market"] = df["market"].str.strip().str.title()

# Parse date and extract month (dayfirst=True for DD-MM-YYYY format)
df["date"]  = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])
df["month"] = df["date"].dt.month

# Keep required columns only
df = df[["crop", "state", "market", "price", "month"]]

# Remove zero or negative prices
df = df[df["price"] > 0]

# Save cleaned data
df.to_csv("cleaned_data.csv", index=False)

print("")
print("cleaned_data.csv saved successfully!")
print("Total records :", len(df))
print("Unique crops  :", sorted(df["crop"].unique().tolist()))
print("Unique states :", sorted(df["state"].unique().tolist()))
print("Unique markets:", len(df["market"].unique()), "markets")
print("")
print("Now run: python train_model.py")