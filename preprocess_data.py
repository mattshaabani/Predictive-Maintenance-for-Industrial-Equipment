import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

DATA_PATH = "data/raw/ai4i2020.csv"
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["UDI", "Product ID"])

print("Missing values:\n", df.isnull().sum())

X = df.drop(columns=["Machine failure"])
y = df["Machine failure"]

numerical_cols = ["Air temperature [K]", "Process temperature [K]", "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]"]
categorical_cols = ["Type"]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(drop="first"), categorical_cols)
    ])

X_processed = preprocessor.fit_transform(X)

joblib.dump(preprocessor, "models/preprocessor.joblib")

X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42, stratify=y)

pd.DataFrame(X_train).to_csv("data/processed/X_train.csv", index=False)
pd.DataFrame(X_test).to_csv("data/processed/X_test.csv", index=False)
pd.Series(y_train, name="Machine failure").to_csv("data/processed/y_train.csv", index=False)
pd.Series(y_test, name="Machine failure").to_csv("data/processed/y_test.csv", index=False)

print("Data preprocessing completed. Files saved to data/processed/")