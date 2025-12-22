import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn
import os

data_path = data_path = "MLProject/hotelbookings_preprocessing/hotelbookings_preprocessing_automate.csv"


if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {data_path}")

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

if "is_canceled" not in df.columns:
    raise ValueError("Kolom 'is_canceled' tidak ditemukan di dataset. Cek header CSV!")

X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    random_state=42,
    n_estimators=100
)

mlflow.set_tracking_uri("file:mlruns")
mlflow.set_experiment("Hotel_Cancellation_Model_Basic")
mlflow.sklearn.autolog()

with mlflow.start_run():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")

print("Training selesai. Folder MLflow runs ada di 'MLProject/mlruns'.")
