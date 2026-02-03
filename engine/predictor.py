
import joblib
import os
import numpy as np

# --- LOAD MODELS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_PATHS = {
    "hr": "heart_rate_model.pkl",
    "rr": "respiration_rate_model.pkl",
    "sys": "systolic_bp_model.pkl",
    "dia": "diastolic_bp_model.pkl",
}

models = {}

print(f"[Predictor] Loading models from {MODELS_DIR}...")
for k, filename in MODEL_PATHS.items():
    p = os.path.join(MODELS_DIR, filename)
    try:
        models[k] = joblib.load(p)
        print(f"Loaded model {k} from {p}")
    except Exception as e:
        print(f"Warning: could not load model {k} from {p}: {e}")
        models[k] = None

def prepare_feature_for_model(model, feat8):
    """
    Adapts the 8-dim feature vector to the model's expected input dimension.
    If the model expects 3 features, it slices the first 3 (mean, std, range).
    """
    try:
        # try to inspect model input dimension (sklearn often has coef_ or n_features_in_)
        n_in = None
        if hasattr(model, "n_features_in_"):
            n_in = model.n_features_in_
        elif hasattr(model, "coef_"):
            n_in = model.coef_.shape[-1]
        
        if n_in == 3:
            # fallback to mean/std/range
            return feat8[:, :3]
    except Exception:
        pass
    return feat8

def predict_value(model_key, features):
    """
    Generic prediction wrapper.
    """
    model = models.get(model_key)
    if model is None:
        return np.nan
    
    try:
        X = prepare_feature_for_model(model, features)
        return float(model.predict(X)[0])
    except Exception as e:
        # print(f"Prediction error for {model_key}: {e}")
        return np.nan

def predict_metrics(features):
    return {
        'hr': predict_value('hr', features),
        'rr': predict_value('rr', features),
        'sys': predict_value('sys', features),
        'dia': predict_value('dia', features)
    }
