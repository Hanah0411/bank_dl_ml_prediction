import os
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__, template_folder='templates')
CORS(app)

DB_PATH = 'predictions.db'
MODEL_PATH = 'modelo_banking.pkl'
FEATURES_PATH = 'feature_columns.json'
MODEL_DL_PATH = 'modelo_dl.h5'
PREPROCESSOR_DL_PATH = 'preprocessor_dl.pkl'
METRICS_PATH = 'metrics.json'

# --- Cargar modelos ---
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
model_dl = load_model(MODEL_DL_PATH) if os.path.exists(MODEL_DL_PATH) else None
preprocessor_dl = joblib.load(PREPROCESSOR_DL_PATH) if os.path.exists(PREPROCESSOR_DL_PATH) else None

# --- Cargar columnas de features ---
feature_columns = []
if os.path.exists(FEATURES_PATH):
    with open(FEATURES_PATH, 'r', encoding='utf-8') as f:
        feature_columns = json.load(f)

# --- Inicializar DB ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        input_json TEXT,
        prediction INTEGER,
        probability REAL,
        model_type TEXT,
        accuracy REAL,
        precision REAL,
        recall REAL,
        f1 REAL,
        confusion_png TEXT,
        roc_png TEXT,
        pr_png TEXT
    )''')
    conn.commit()
    conn.close()

init_db()

def _prepare_input_df(user_input):
    return pd.DataFrame([user_input], columns=feature_columns)

# --- Endpoints ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

@app.route('/history_data')
def history_data():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, timestamp, input_json, prediction, probability, model_type, accuracy, precision, recall, f1, confusion_png, roc_png, pr_png FROM predictions ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()

    data = []
    for r in rows:
        try:
            input_data = json.loads(r[2])
        except:
            input_data = {}
        data.append({
            "id": r[0],
            "timestamp": r[1],
            "input": input_data,
            "prediction": r[3],
            "probability": r[4],
            "model_type": r[5],
            "accuracy": r[6],
            "precision": r[7],
            "recall": r[8],
            "f1": r[9],
            "confusion_png": r[10] or "",
            "roc_png": r[11] or "",
            "pr_png": r[12] or ""
        })
    return jsonify(data)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or "input" not in data or "model_type" not in data:
        return jsonify({"error": "Se requiere 'input' y 'model_type'"}), 400

    user_input = data["input"]
    model_type = data["model_type"].lower()

    numeric_fields = ['age','balance','day','duration','campaign','pdays','previous']
    for f in numeric_fields:
        if f in user_input and user_input[f] != '':
            try: user_input[f] = float(user_input[f])
            except: user_input[f] = 0.0

    X_input = _prepare_input_df(user_input)

    try:
        if model_type == "dl":
            if not model_dl or not preprocessor_dl:
                return jsonify({"error": "⚠ Modelo DL no cargado"}), 500
            X_processed = preprocessor_dl.transform(X_input.astype(str)).astype(np.float32)
            prob = float(model_dl.predict(X_processed)[0][0])
            prediction = 1 if prob >= 0.5 else 0
        else:  # ML
            if not model: return jsonify({"error": "⚠ Modelo ML no cargado"}), 500
            prob = float(model.predict_proba(X_input)[0][1])
            prediction = int(model.predict(X_input)[0])

        # --- Generar métricas (dummy, puedes mejorar según tu evaluación) ---
        accuracy = 0.85
        precision = 0.82
        recall = 0.8
        f1 = 0.81

        # --- Generar gráficas de ejemplo ---
        def plot_to_base64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)
            return f"data:image/png;base64,{img_base64}"

        # Matriz de confusión de ejemplo
        fig1, ax1 = plt.subplots()
        ax1.imshow([[50,10],[5,35]], cmap='Blues')
        ax1.set_title("Confusion Matrix")
        confusion_png = plot_to_base64(fig1)

        # ROC dummy
        fig2, ax2 = plt.subplots()
        ax2.plot([0,0.2,0.5,1],[0,0.6,0.8,1], color='red')
        ax2.set_title("ROC Curve")
        roc_png = plot_to_base64(fig2)

        # PR dummy
        fig3, ax3 = plt.subplots()
        ax3.plot([0,0.2,0.5,1],[1,0.8,0.6,0], color='green')
        ax3.set_title("Precision-Recall")
        pr_png = plot_to_base64(fig3)

        # Guardar en DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            """INSERT INTO predictions 
            (timestamp, input_json, prediction, probability, model_type, accuracy, precision, recall, f1, confusion_png, roc_png, pr_png) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now().isoformat(), json.dumps(user_input), prediction, prob, model_type,
             accuracy, precision, recall, f1, confusion_png, roc_png, pr_png)
        )
        conn.commit()
        conn.close()

        return jsonify({
            "prediction": prediction,
            "probability": prob,
            "used_model": model_type,
            "message": "Cliente responde positivamente" if prediction==1 else "Cliente NO responde"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/metrics')
def metrics_endpoint():
    metrics = {"accuracy":0, "precision":0, "recall":0, "f1":0}
    model_type = request.args.get('model', 'ml').lower()
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, 'r', encoding='utf-8') as f:
                file_metrics = json.load(f)
                if model_type in file_metrics:
                    metrics = {k: float(file_metrics[model_type].get(k,0)) for k in metrics.keys()}
        except Exception as e:
            print(f"Error leyendo métricas: {e}")
    return jsonify(metrics)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


