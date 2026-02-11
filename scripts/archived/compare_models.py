from pathlib import Path
import joblib

model_paths = [
    Path("ml_data/best_model_with_spreads.joblib"),
    Path("ml_data/best_model.joblib"),
    Path("ml_data/best_model_optimized.joblib"),
    Path("ml_data/ridge_model.joblib"),
]

info = {}
for path in model_paths:
    if not path.exists():
        continue
    model = joblib.load(path)
    params = model.get_params() if hasattr(model, "get_params") else {}
    info[path.name] = {
        "type": type(model).__name__,
        "params": params,
    }

print(info)
