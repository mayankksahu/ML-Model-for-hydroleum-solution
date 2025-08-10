import json
import requests
import joblib
import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS

THINGSPEAK_URL = "https://api.thingspeak.com/channels/3024727/feeds.json?results=1"
LOCAL_TANKDATA_FILE = "tank1_dummy_data_15kl_1000.json"

MODEL_PATH = "models/threat_model.pkl"
IMPUTER_PATH = "models/imputer.pkl"

app = Flask(__name__)
CORS(app)

# Load model + imputer
try:
    model = joblib.load(MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
except Exception as e:
    print(f"[ERROR] Could not load model or imputer: {e}")
    model = None
    imputer = None


def fetch_circuit_vitals():
    try:
        resp = requests.get(THINGSPEAK_URL, timeout=5)
        resp.raise_for_status()
        data = resp.json()

        if "feeds" not in data or not data["feeds"]:
            raise ValueError("No feeds in ThingSpeak data")

        latest = data["feeds"][-1]
        water_level_percent = float(latest.get("field1", 0)) / 100

        return {
            "water_level_percent": float(round(water_level_percent, 2)),
            "float_sensor": int(latest.get("field2", 0)),
            "pump_state": int(latest.get("field3", 0))
        }
    except Exception as e:
        return {"error": f"CircuitData fetch failed: {e}"}


def fetch_tankdata():
    try:
        with open(LOCAL_TANKDATA_FILE, "r") as f:
            data = json.load(f)
        latest = data[0] if isinstance(data, list) else data

        return {
            "capacity": float(latest.get("capacity", 0)),
            "temperature_C": float(latest.get("temperature_C", 0)),
            "ullage_ltr": float(latest.get("ullage_ltr", 0))
        }
    except Exception as e:
        return {"error": f"TankData fetch failed: {e}"}


def rule_based_check(circuit, tank):
    reasons = []
    threat_level = "Normal"

    if circuit.get("water_level_percent", 0) > 0.05:
        reasons.append("Water level above safe limit")
    if circuit.get("float_sensor", 0) == 1:
        reasons.append("Float sensor triggered")
    if tank.get("temperature_C", 0) > 50:
        reasons.append("Tank temperature too high")
    if tank.get("ullage_ltr", 0) < 1000:
        reasons.append("Low ullage — risk of overflow")

    if len(reasons) >= 2:
        threat_level = "Critical"
    elif len(reasons) == 1:
        threat_level = "Warning"

    return threat_level, reasons


def ml_predict(circuit, tank):
    if not model or not imputer:
        return None, "ML model or imputer not loaded"

    try:
        features = np.array([[
            circuit.get("water_level_percent", 0),
            circuit.get("float_sensor", 0),
            circuit.get("pump_state", 0),
            tank.get("capacity", 0),
            tank.get("temperature_C", 0),
            tank.get("ullage_ltr", 0)
        ]], dtype=float)

        features = imputer.transform(features)
        pred = model.predict(features)[0]
        return str(pred), None  # Ensure it's JSON serializable
    except Exception as e:
        return None, f"ML prediction failed: {e}"


@app.route("/api/threat", methods=["GET"])
def threat_output():
    circuit_data = fetch_circuit_vitals()
    tank_data = fetch_tankdata()

    if "error" in circuit_data:
        return jsonify({"error": circuit_data["error"]}), 500
    if "error" in tank_data:
        return jsonify({"error": tank_data["error"]}), 500

    # Convert all values to Python native types
    circuit_data = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in circuit_data.items()}
    tank_data = {k: float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v for k, v in tank_data.items()}

    rule_threat, reasons = rule_based_check(circuit_data, tank_data)

    ml_pred, ml_error = ml_predict(circuit_data, tank_data)
    if ml_error:
        reasons.append(ml_error)

    final_threat = rule_threat
    if ml_pred == "Critical" or rule_threat == "Critical":
        final_threat = "Critical"
    elif ml_pred == "Warning" or rule_threat == "Warning":
        final_threat = "Warning"

    return jsonify({
        "threat_level": str(final_threat),
        "reason": "; ".join(str(r) for r in reasons) if reasons else "All parameters normal",
        "ml_prediction": ml_pred,
        "readings": {**circuit_data, **tank_data},
        "rules_triggered": [str(r) for r in reasons]
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
