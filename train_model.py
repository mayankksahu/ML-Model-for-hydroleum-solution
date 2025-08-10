import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# ====== LOAD DATA ======
df = pd.read_json("tank1_dummy_data_15kl_1000.json")

# --- Define target mapping rules ---
def map_threat(row):
    # Example: adjust according to your domain rules
    if (row["water_ltr"] > 500) or (row["temperature_C"] > 50):
        return 2  # Critical
    elif (row["water_ltr"] > 200) or (row["temperature_C"] > 40):
        return 1  # Warning
    else:
        return 0  # Safe

# Apply mapping
df["threat_level"] = df.apply(map_threat, axis=1)

# ====== FEATURE SELECTION ======
features = [
    "water_ltr",
    "temperature_C",
    "ullage_ltr",
    "netVolume_ltr",
    "todaysSale_ltr",
    "todaysDelivery_ltr"
]

X = df[features]
y = df["threat_level"].astype(int)

# ====== SPLIT ======
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ====== PIPELINE ======
pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("clf", RandomForestClassifier(n_estimators=200, random_state=42))
])

# ====== TRAIN ======
pipeline.fit(X_train, y_train)

# ====== EVALUATE ======
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# ====== SAVE ======
import os
os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/threat_model.pkl")
joblib.dump(pipeline.named_steps["imputer"], "models/imputer.pkl")

print("✅ Model retrained and saved. All predictions mapped to 0/1/2.")
