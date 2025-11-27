# train_dl.py
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Leer dataset
df = pd.read_csv("bank.csv")
X = df.drop("deposit", axis=1)
y = df["deposit"].map({"yes": 1, "no": 0})

# Columnas
cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# Preprocesador
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ("num", "passthrough", num_cols)
])

X_processed = preprocessor.fit_transform(X)

# Modelo DL simple
model_dl = Sequential([
    Dense(32, input_shape=(X_processed.shape[1],), activation="relu"),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

model_dl.compile(optimizer=Adam(0.001), loss="binary_crossentropy", metrics=["accuracy"])
model_dl.fit(X_processed, y, epochs=20, batch_size=32, verbose=1)

# Guardar modelo y preprocesador
model_dl.save("modelo_dl.h5")
joblib.dump(preprocessor, "preprocessor_dl.pkl")
print("Modelo DL y preprocesador guardados ✔")

