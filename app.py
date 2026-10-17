from pathlib import Path
from flask import Flask, jsonify, render_template, request
from model import Predictor

ROOT = Path(__file__).resolve().parent
app = Flask(__name__)
predictor = Predictor(ROOT / "model_artifacts")

@app.get("/")
def index(): return render_template("index.html")

@app.get("/api/schema")
def schema(): return jsonify(predictor.schema())

@app.get("/api/health")
def health(): return jsonify({"status":"ok", "models":len(predictor.models), "seeds":predictor.seeds})

@app.post("/api/predict")
def predict():
    try:
        return jsonify(predictor.predict(request.get_json(force=True)))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8501, debug=False)
