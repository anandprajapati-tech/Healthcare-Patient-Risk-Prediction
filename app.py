"""
Healthcare Patient Risk Prediction - Flask Backend Application
Provides ML-based disease risk estimation for:
- Diabetes
- Heart Disease
- Chronic Kidney Disease
- Liver Disease
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("HealthcareRiskApp")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Load Models, Preprocessors, and Metadata once at startup
MODELS = {}
PREPROCESSORS = {}
METADATA = {}

try:
    for disease in ["diabetes", "heart", "kidney", "liver"]:
        model_path = os.path.join(MODELS_DIR, f"{disease}_model.pkl")
        prep_path = os.path.join(MODELS_DIR, f"{disease}_preprocessor.pkl")
        
        if os.path.exists(model_path) and os.path.exists(prep_path):
            MODELS[disease] = joblib.load(model_path)
            PREPROCESSORS[disease] = joblib.load(prep_path)
            logger.info(f"Loaded model and preprocessor for: {disease}")
        else:
            logger.warning(f"Model or preprocessor missing for: {disease} (expected in {MODELS_DIR})")
            
    meta_path = os.path.join(MODELS_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            METADATA = json.load(f)
            logger.info("Loaded model metadata successfully.")
except Exception as e:
    logger.error(f"Error loading models at startup: {e}")


# Define expected features and friendly descriptions for each disease
DISEASE_CONFIG = {
    "diabetes": {
        "display_name": "Diabetes",
        "algorithm": "Random Forest Classifier",
        "features": [
            {"name": "Pregnancies", "label": "Number of Pregnancies", "type": "number", "min": 0, "max": 20, "step": 1, "default": 2},
            {"name": "Glucose", "label": "Glucose (mg/dL)", "type": "number", "min": 0, "max": 300, "step": 1, "default": 120},
            {"name": "BloodPressure", "label": "Diastolic Blood Pressure (mm Hg)", "type": "number", "min": 0, "max": 180, "step": 1, "default": 70},
            {"name": "SkinThickness", "label": "Triceps Skin Fold (mm)", "type": "number", "min": 0, "max": 100, "step": 1, "default": 20},
            {"name": "Insulin", "label": "2-Hour Serum Insulin (mu U/ml)", "type": "number", "min": 0, "max": 900, "step": 1, "default": 80},
            {"name": "BMI", "label": "Body Mass Index (BMI kg/m²)", "type": "number", "min": 10.0, "max": 70.0, "step": 0.1, "default": 26.5},
            {"name": "DiabetesPedigreeFunction", "label": "Diabetes Pedigree Function", "type": "number", "min": 0.01, "max": 2.5, "step": 0.01, "default": 0.45},
            {"name": "Age", "label": "Age (years)", "type": "number", "min": 1, "max": 120, "step": 1, "default": 35}
        ]
    },
    "heart": {
        "display_name": "Heart Disease",
        "algorithm": "Random Forest Classifier",
        "features": [
            {"name": "age", "label": "Age (years)", "type": "number", "min": 1, "max": 120, "step": 1, "default": 55},
            {"name": "sex", "label": "Sex", "type": "select", "options": [{"value": "1", "label": "Male"}, {"value": "0", "label": "Female"}], "default": "1"},
            {"name": "cp", "label": "Chest Pain Type", "type": "select", "options": [
                {"value": "0", "label": "Typical Angina (0)"},
                {"value": "1", "label": "Atypical Angina (1)"},
                {"value": "2", "label": "Non-anginal Pain (2)"},
                {"value": "3", "label": "Asymptomatic (3)"}
            ], "default": "1"},
            {"name": "trestbps", "label": "Resting Blood Pressure (mm Hg)", "type": "number", "min": 60, "max": 240, "step": 1, "default": 130},
            {"name": "chol", "label": "Serum Cholesterol (mg/dl)", "type": "number", "min": 100, "max": 600, "step": 1, "default": 230},
            {"name": "fbs", "label": "Fasting Blood Sugar > 120 mg/dl", "type": "select", "options": [{"value": "0", "label": "No (<= 120 mg/dl)"}, {"value": "1", "label": "Yes (> 120 mg/dl)"}], "default": "0"},
            {"name": "restecg", "label": "Resting ECG Results", "type": "select", "options": [
                {"value": "0", "label": "Normal (0)"},
                {"value": "1", "label": "ST-T Wave Abnormality (1)"},
                {"value": "2", "label": "Left Ventricular Hypertrophy (2)"}
            ], "default": "0"},
            {"name": "thalach", "label": "Max Heart Rate Achieved (bpm)", "type": "number", "min": 60, "max": 230, "step": 1, "default": 150},
            {"name": "exang", "label": "Exercise Induced Angina", "type": "select", "options": [{"value": "0", "label": "No"}, {"value": "1", "label": "Yes"}], "default": "0"},
            {"name": "oldpeak", "label": "ST Depression (Oldpeak)", "type": "number", "min": 0.0, "max": 10.0, "step": 0.1, "default": 1.0},
            {"name": "slope", "label": "Slope of Peak Exercise ST", "type": "select", "options": [
                {"value": "0", "label": "Upsloping (0)"},
                {"value": "1", "label": "Flat (1)"},
                {"value": "2", "label": "Downsloping (2)"}
            ], "default": "1"},
            {"name": "ca", "label": "Major Vessels Colored (0-4)", "type": "select", "options": [
                {"value": "0", "label": "0"}, {"value": "1", "label": "1"},
                {"value": "2", "label": "2"}, {"value": "3", "label": "3"}, {"value": "4", "label": "4"}
            ], "default": "0"},
            {"name": "thal", "label": "Thalassemia", "type": "select", "options": [
                {"value": "1", "label": "Normal (1)"},
                {"value": "2", "label": "Fixed Defect (2)"},
                {"value": "3", "label": "Reversible Defect (3)"},
                {"value": "0", "label": "Other (0)"}
            ], "default": "2"}
        ]
    },
    "kidney": {
        "display_name": "Chronic Kidney Disease",
        "algorithm": "Random Forest Classifier",
        "features": [
            # Numerical
            {"name": "age", "label": "Age (years)", "type": "number", "min": 1, "max": 120, "step": 1, "default": 48},
            {"name": "bp", "label": "Blood Pressure (mm Hg)", "type": "number", "min": 40, "max": 200, "step": 1, "default": 80},
            {"name": "sg", "label": "Specific Gravity", "type": "select", "options": [
                {"value": "1.005", "label": "1.005"}, {"value": "1.010", "label": "1.010"},
                {"value": "1.015", "label": "1.015"}, {"value": "1.020", "label": "1.020"},
                {"value": "1.025", "label": "1.025"}
            ], "default": "1.020"},
            {"name": "al", "label": "Albumin", "type": "select", "options": [
                {"value": "0", "label": "0 (Normal)"}, {"value": "1", "label": "1"},
                {"value": "2", "label": "2"}, {"value": "3", "label": "3"},
                {"value": "4", "label": "4"}, {"value": "5", "label": "5"}
            ], "default": "0"},
            {"name": "su", "label": "Sugar", "type": "select", "options": [
                {"value": "0", "label": "0 (Normal)"}, {"value": "1", "label": "1"},
                {"value": "2", "label": "2"}, {"value": "3", "label": "3"},
                {"value": "4", "label": "4"}, {"value": "5", "label": "5"}
            ], "default": "0"},
            {"name": "bgr", "label": "Blood Glucose Random (mg/dl)", "type": "number", "min": 40, "max": 500, "step": 1, "default": 110},
            {"name": "bu", "label": "Blood Urea (mg/dl)", "type": "number", "min": 5, "max": 400, "step": 0.1, "default": 35},
            {"name": "sc", "label": "Serum Creatinine (mg/dl)", "type": "number", "min": 0.1, "max": 20.0, "step": 0.1, "default": 1.1},
            {"name": "sod", "label": "Sodium (mEq/L)", "type": "number", "min": 100, "max": 180, "step": 0.1, "default": 138},
            {"name": "pot", "label": "Potassium (mEq/L)", "type": "number", "min": 1.0, "max": 15.0, "step": 0.1, "default": 4.2},
            {"name": "hemo", "label": "Hemoglobin (g/dl)", "type": "number", "min": 3.0, "max": 20.0, "step": 0.1, "default": 14.5},
            {"name": "pcv", "label": "Packed Cell Volume (%)", "type": "number", "min": 10, "max": 60, "step": 1, "default": 42},
            {"name": "wc", "label": "White Blood Cell Count (/cumm)", "type": "number", "min": 1500, "max": 30000, "step": 100, "default": 7500},
            {"name": "rc", "label": "Red Blood Cell Count (millions/cmm)", "type": "number", "min": 1.0, "max": 9.0, "step": 0.1, "default": 4.8},
            # Categorical
            {"name": "rbc", "label": "Red Blood Cells", "type": "select", "options": [{"value": "normal", "label": "Normal"}, {"value": "abnormal", "label": "Abnormal"}], "default": "normal"},
            {"name": "pc", "label": "Pus Cell", "type": "select", "options": [{"value": "normal", "label": "Normal"}, {"value": "abnormal", "label": "Abnormal"}], "default": "normal"},
            {"name": "pcc", "label": "Pus Cell Clumps", "type": "select", "options": [{"value": "notpresent", "label": "Not Present"}, {"value": "present", "label": "Present"}], "default": "notpresent"},
            {"name": "ba", "label": "Bacteria", "type": "select", "options": [{"value": "notpresent", "label": "Not Present"}, {"value": "present", "label": "Present"}], "default": "notpresent"},
            {"name": "htn", "label": "Hypertension", "type": "select", "options": [{"value": "no", "label": "No"}, {"value": "yes", "label": "Yes"}], "default": "no"},
            {"name": "dm", "label": "Diabetes Mellitus", "type": "select", "options": [{"value": "no", "label": "No"}, {"value": "yes", "label": "Yes"}], "default": "no"},
            {"name": "cad", "label": "Coronary Artery Disease", "type": "select", "options": [{"value": "no", "label": "No"}, {"value": "yes", "label": "Yes"}], "default": "no"},
            {"name": "appet", "label": "Appetite", "type": "select", "options": [{"value": "good", "label": "Good"}, {"value": "poor", "label": "Poor"}], "default": "good"},
            {"name": "pe", "label": "Pedal Edema", "type": "select", "options": [{"value": "no", "label": "No"}, {"value": "yes", "label": "Yes"}], "default": "no"},
            {"name": "ane", "label": "Anemia", "type": "select", "options": [{"value": "no", "label": "No"}, {"value": "yes", "label": "Yes"}], "default": "no"}
        ]
    },
    "liver": {
        "display_name": "Liver Disease",
        "algorithm": "Logistic Regression (balanced)",
        "features": [
            {"name": "Age", "label": "Age (years)", "type": "number", "min": 1, "max": 120, "step": 1, "default": 45},
            {"name": "Gender", "label": "Gender", "type": "select", "options": [{"value": "Male", "label": "Male"}, {"value": "Female", "label": "Female"}], "default": "Male"},
            {"name": "Total_Bilirubin", "label": "Total Bilirubin (mg/dL)", "type": "number", "min": 0.1, "max": 80.0, "step": 0.1, "default": 0.9},
            {"name": "Direct_Bilirubin", "label": "Direct Bilirubin (mg/dL)", "type": "number", "min": 0.05, "max": 40.0, "step": 0.05, "default": 0.25},
            {"name": "Alkaline_Phosphotase", "label": "Alkaline Phosphatase (IU/L)", "type": "number", "min": 20, "max": 2500, "step": 1, "default": 190},
            {"name": "Alamine_Aminotransferase", "label": "ALT / SGPT (IU/L)", "type": "number", "min": 5, "max": 2500, "step": 1, "default": 30},
            {"name": "Aspartate_Aminotransferase", "label": "AST / SGOT (IU/L)", "type": "number", "min": 5, "max": 5000, "step": 1, "default": 35},
            {"name": "Total_Protiens", "label": "Total Proteins (g/dL)", "type": "number", "min": 1.0, "max": 12.0, "step": 0.1, "default": 6.8},
            {"name": "Albumin", "label": "Albumin (g/dL)", "type": "number", "min": 0.5, "max": 8.0, "step": 0.1, "default": 3.6},
            {"name": "Albumin_and_Globulin_Ratio", "label": "Albumin and Globulin Ratio", "type": "number", "min": 0.1, "max": 3.5, "step": 0.01, "default": 1.0}
        ]
    }
}


@app.route("/", methods=["GET"])
def index():
    """Renders the main healthcare risk prediction interface."""
    return render_template("index.html", disease_config=DISEASE_CONFIG, metadata=METADATA)


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "models_loaded": list(MODELS.keys())
    }), 200


@app.route("/model-info", methods=["GET"])
def model_info():
    """Returns metadata and actual performance metrics of all 4 disease models."""
    return jsonify({
        "status": "success",
        "models": METADATA,
        "config": {
            k: {
                "display_name": v["display_name"],
                "algorithm": v["algorithm"],
                "features_count": len(v["features"])
            }
            for k, v in DISEASE_CONFIG.items()
        }
    }), 200


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predicts disease risk using the trained ML model for the selected disease.
    Accepts JSON or Form-encoded data.
    """
    try:
        # 1. Parse request payload
        if request.is_json:
            payload = request.get_json() or {}
        else:
            payload = request.form.to_dict()

        if not payload:
            return jsonify({
                "status": "error",
                "message": "Empty request payload. Please submit valid patient data."
            }), 400

        # 2. Extract patient name (display only - never used as a model feature, never stored)
        patient_name = str(payload.get("patient_name", "")).strip()
        if not patient_name:
            patient_name = "Anonymous Patient"

        # 3. Identify and validate disease
        disease = str(payload.get("disease", "")).strip().lower()
        if not disease or disease not in DISEASE_CONFIG:
            return jsonify({
                "status": "error",
                "message": f"Unsupported or missing disease. Please select from: {list(DISEASE_CONFIG.keys())}"
            }), 400

        # 4. Check model availability
        if disease not in MODELS or disease not in PREPROCESSORS:
            return jsonify({
                "status": "error",
                "message": f"Trained model for {disease.title()} is currently unavailable on the server."
            }), 500

        model = MODELS[disease]
        preprocessor = PREPROCESSORS[disease]
        cfg = DISEASE_CONFIG[disease]

        # 5. Extract and validate required features
        input_data = {}
        missing_fields = []
        validation_errors = []

        for feat in cfg["features"]:
            fname = feat["name"]
            flabel = feat["label"]
            ftype = feat["type"]

            val = payload.get(fname)

            if val is None or str(val).strip() == "":
                missing_fields.append(f"{flabel} ({fname})")
                continue

            # Convert and validate numeric inputs
            if ftype == "number":
                try:
                    num_val = float(val)
                    min_val = feat.get("min")
                    max_val = feat.get("max")
                    if min_val is not None and num_val < min_val:
                        validation_errors.append(f"{flabel} must be at least {min_val}.")
                    elif max_val is not None and num_val > max_val:
                        validation_errors.append(f"{flabel} cannot exceed {max_val}.")
                    else:
                        input_data[fname] = num_val
                except ValueError:
                    validation_errors.append(f"{flabel} must be a valid number.")
            else:
                # Select / categorical
                s_val = str(val).strip()
                valid_options = [opt["value"] for opt in feat.get("options", [])]
                if valid_options and s_val not in valid_options:
                    validation_errors.append(f"Invalid option for {flabel}.")
                else:
                    input_data[fname] = s_val

        if missing_fields:
            return jsonify({
                "status": "error",
                "message": f"Missing required fields: {', '.join(missing_fields[:3])}" + ("..." if len(missing_fields) > 3 else "")
            }), 400

        if validation_errors:
            return jsonify({
                "status": "error",
                "message": " | ".join(validation_errors[:2])
            }), 400

        # 6. Build DataFrame for preprocessor
        row_df = pd.DataFrame([input_data])

        # Apply specific domain adjustments (e.g. biological zeros in diabetes)
        if disease == "diabetes":
            zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in zero_cols:
                if col in row_df.columns:
                    row_df[col] = row_df[col].replace(0, np.nan)

        # 7. Preprocess features using saved pipeline
        try:
            X_proc = preprocessor.transform(row_df)
        except Exception as prep_err:
            logger.error(f"Preprocessing error for {disease}: {prep_err}")
            return jsonify({
                "status": "error",
                "message": "Error preprocessing clinical features. Please verify input values."
            }), 400

        # 8. Generate actual model prediction
        pred_raw = int(model.predict(X_proc)[0])

        # 9. Compute probability if model supports predict_proba
        probability = None
        confidence_text = None
        if hasattr(model, "predict_proba"):
            try:
                proba_array = model.predict_proba(X_proc)[0]
                # Positive risk probability (class index 1)
                prob_pct = round(float(proba_array[1]) * 100, 2)
                probability = prob_pct
                confidence_text = f"{prob_pct}% estimated risk"
            except Exception as prob_err:
                logger.warning(f"Could not compute predict_proba for {disease}: {prob_err}")
                probability = None

        # 10. Format clinical labels and explanation
        if pred_raw == 1:
            risk_level = "Elevated Risk Detected"
            risk_class = "risk-high"
            summary = (
                f"Based on the provided clinical parameters, the {cfg['algorithm']} model "
                f"classifies {patient_name} as having an ELEVATED risk for {cfg['display_name']}."
            )
        else:
            risk_level = "Low / Normal Risk"
            risk_class = "risk-low"
            summary = (
                f"Based on the provided clinical parameters, the {cfg['algorithm']} model "
                f"classifies {patient_name} within the LOW / NORMAL risk category for {cfg['display_name']}."
            )

        response_payload = {
            "status": "success",
            "patient_name": patient_name,
            "disease_id": disease,
            "disease_name": cfg["display_name"],
            "prediction": pred_raw,
            "risk_level": risk_level,
            "risk_class": risk_class,
            "probability": probability,
            "confidence_text": confidence_text,
            "algorithm": cfg["algorithm"],
            "summary": summary,
            "status_label": "Educational ML Risk Estimate",
            "disclaimer": "Educational ML estimate only — not a medical diagnosis."
        }

        return jsonify(response_payload), 200

    except Exception as e:
        logger.exception(f"Unexpected server error during prediction: {e}")
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred while processing the prediction. Please try again."
        }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting Healthcare Risk Prediction App on port {port}...")
    app.run(host="0.0.0.0", port=port, debug=False)
