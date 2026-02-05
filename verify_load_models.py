import os
import traceback
import joblib

root = os.path.dirname(__file__)
ml_dir = os.path.join(root, "ml_data")

files = [f for f in os.listdir(ml_dir) if f.endswith('.joblib')]
if not files:
    print('No .joblib files found in', ml_dir)
    raise SystemExit(1)

for fn in files:
    path = os.path.join(ml_dir, fn)
    print('\n---', fn)
    try:
        obj = joblib.load(path)
        print('Loaded OK; type:', type(obj))
        # Common attributes to inspect
        for attr in ('coef_', 'feature_importances_', 'n_features_in_', 'classes_', 'score'):
            if hasattr(obj, attr):
                try:
                    val = getattr(obj, attr)
                    print(f"  {attr}:", type(val), getattr(val, '__len__', lambda: None)())
                except Exception:
                    print(f"  {attr}: <error reading>")
    except Exception:
        print('ERROR loading', fn)
        traceback.print_exc()

print('\nDone')
