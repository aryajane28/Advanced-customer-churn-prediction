from src.data_preprocessing import load_data, preprocess
from src.train_model import train_models

df = load_data("data/raw/telco_churn.csv")
X_train, X_test, y_train, y_test = preprocess(df)

train_models(X_train, X_test, y_train, y_test)