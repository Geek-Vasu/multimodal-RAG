import pandas as pd
import os

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH=os.path.join(BASE_DIR,"data","product_metadata.csv")

df=pd.read_csv(CSV_PATH)

def search_by_metadata(filters:dict):
    filtered=df.copy()

    for key, value in filters.items():
        if value is None:
            continue
        filtered=filtered[filtered[key].str.lower()==value.lower()]

    return filtered.to_dict(orient="records")