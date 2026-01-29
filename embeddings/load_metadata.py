import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "product_metadata.csv")

def load_metadata():
    df = pd.read_csv(CSV_PATH)
    return df

if __name__ == "__main__":
    df = load_metadata()
    print("Metadata loaded successfully:")
    print(df.head())
