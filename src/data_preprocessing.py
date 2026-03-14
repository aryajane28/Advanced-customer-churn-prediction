import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess(df):

    df = df.dropna()

    encoder = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    return train_test_split(X, y, test_size=0.2, random_state=42)