"""
XGBoost inference — called by server.js via child_process.
Reads features from stdin as JSON, writes prediction to stdout as JSON.
Usage: echo '{"symbol":"BTC","features":{...}}' | python ml_predict.py
"""
import sys, json, os
import numpy as np
from xgboost import XGBClassifier

def load_model(symbol):
    model_path = os.path.join(os.path.dirname(__file__), f'model_{symbol}.json')
    meta_path  = os.path.join(os.path.dirname(__file__), f'model_{symbol}_meta.json')
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
    model = XGBClassifier()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta

def predict(symbol, features_dict):
    model, meta = load_model(symbol)
    if model is None:
        return {'error': 'model_not_found', 'signal': None, 'confidence': 0,
                'probs': {'UP': 0.33, 'DOWN': 0.33, 'SIDEWAYS': 0.34}}

    if meta.get('version') == 'insufficient':
        return {'error': 'insufficient_data', 'n_samples': meta.get('n_samples', 0),
                'signal': None, 'confidence': 0,
                'probs': {'UP': 0.33, 'DOWN': 0.33, 'SIDEWAYS': 0.34}}

    feature_cols = meta['feature_cols']
    classes = meta['classes']  # e.g. ['DOWN','SIDEWAYS','UP']

    # Build feature vector in correct order
    X = np.array([[features_dict.get(c, 0) or 0 for c in feature_cols]], dtype=np.float32)
    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    probs_raw = model.predict_proba(X)[0]
    pred_idx  = int(np.argmax(probs_raw))
    signal    = classes[pred_idx]
    confidence = int(round(float(probs_raw[pred_idx]) * 100))

    probs_dict = {c: round(float(p)*100, 1) for c, p in zip(classes, probs_raw)}

    return {
        'signal': signal,
        'confidence': confidence,
        'probs': probs_dict,
        'model_version': meta.get('version','v1'),
        'wf_acc': meta.get('wf_avg_test_acc', 0),
        'n_samples': meta.get('n_samples', 0),
        'top_features': meta.get('top_features', {})
    }

if __name__ == '__main__':
    try:
        inp = json.loads(sys.stdin.read())
        result = predict(inp['symbol'], inp['features'])
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e), 'signal': None, 'confidence': 0}))
