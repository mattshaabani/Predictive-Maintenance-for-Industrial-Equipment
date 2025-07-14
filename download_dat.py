import pandas as pd
import urllib.request
import os

# Define dataset URL and local path
DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
DATA_PATH = "data/raw/ai4i2020.csv"

# Create data directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

# Download dataset
def download_dataset():
    print("Downloading dataset...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"Dataset downloaded to {DATA_PATH}")

# Load dataset
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset loaded with shape: {df.shape}")
    return df

if __name__ == "__main__":
    download_dataset()
    df = load_dataset()
    print(df.head())