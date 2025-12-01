# train_dl.py - Deep Learning Model Trainer

import pandas as pd
import joblib
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Leer dataset
df = pd.read_csv("bank.csv")
X = df.drop("deposit", axis=1)
y = df["deposit"].map({"yes": 1, "no": 0})

# Columnas categóricas y numéricas
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Preprocesador
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

X_processed = preprocessor.fit_transform(X)

# División en train-test
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Modelo DL
model_dl = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1, activation="sigmoid")
])

model_dl.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Entrenamiento
model_dl.fit(
    X_train, y_train,
    epochs=25,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# Guardar modelo y preprocesador
model_dl.save("modelo_dl.h5")
joblib.dump(preprocessor, "preprocessor_dl.pkl")

# Predicciones + métricas
y_pred_prob = model_dl.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

with open("metrics_dl.json", "w") as f:
    json.dump(metrics, f)

print("\n✔ Modelo DL y preprocesador guardados")
print("📊 Métricas:", metrics)
