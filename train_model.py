# train_model.py (Machine Learning)

import pandas as pd
import joblib
import json
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Leer dataset
df = pd.read_csv("bank.csv")
X = df.drop("deposit", axis=1)
y = df["deposit"].map({"yes": 1, "no": 0})

# Identificar columnas categóricas y numéricas
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Preprocesador para transformar variables
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

# Transformación
X_processed = preprocessor.fit_transform(X)

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)

# Modelo ML
model_ml = RandomForestClassifier(n_estimators=150, random_state=42)
model_ml.fit(X_train, y_train)

# Predicciones y métricas
y_pred = model_ml.predict(X_test)
metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred)
}

# Guardar archivos
joblib.dump(model_ml, "modelo_ml.pkl")
joblib.dump(preprocessor, "preprocessor_ml.pkl")

with open("metrics_ml.json", "w") as f:
    json.dump(metrics, f)

print("Modelo ML entrenado ✔")
print("Metrics:", metrics)

