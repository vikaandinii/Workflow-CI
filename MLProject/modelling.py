import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

data_path = os.path.join(os.path.dirname(__file__), "hotelbookings_preprocessing/hotelbookings_preprocessing_automate.csv")
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset tidak ditemukan: {os.path.abspath(data_path)}")

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

X = df.drop(columns=["is_canceled"])
y = df["is_canceled"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
model = RandomForestClassifier(random_state=42, n_estimators=100)

mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_name = "Hotel_Cancellation_Model_Basic"
mlflow.set_experiment(experiment_name)

with mlflow.start_run(run_name=None, nested=False):
    mlflow.sklearn.autolog()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {acc:.4f}")

print(f"Training selesai. Artefak MLflow disimpan di folder mlruns/ atau database mlflow.db")
