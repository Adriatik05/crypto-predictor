"""
XGBoost inference — called by server.js via child_process.
Reads features from stdin as JSON, writes prediction to stdout as JSON.
Usage: echo '{"symbol":"BTC","features":{...}}' | python ml-predict.py
"""
import sys, json, os
import numpy as np

def load_model(symbol):
    model_path = os.path.join(os.path.dirname(__file__), f'model_{symbol}.json')
    meta_path  = os.path.join(os.path.dirname(__file__), f'model_{symbol}_meta.json')
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        return None, None
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)
    return model, meta

def add_derived_features(features):
    f = dict(features)
    atr = f.get('atr') or 0
    atr_safe = atr if atr else 1e-9
    ema21 = f.get('ema21') or 0
    bb_width = (f.get('bb_upper') or 0) - (f.get('bb_lower') or 0)

    f['ema_spread_pct'] = ((f.get('ema9') or 0) - ema21) / (ema21 or 1e-9) * 100
    f['macd_hist_atr'] = (f.get('macd_hist') or 0) / atr_safe
    f['bb_width_atr'] = bb_width / atr_safe
    f['trend_strength'] = (f.get('adx') or 0) * np.sign((f.get('di_plus') or 0) - (f.get('di_minus') or 0))
    f['ob_imb_avg'] = np.mean([f.get('ob_imb_l5') or 0, f.get('ob_imb_l10') or 0, f.get('ob_imb_l20') or 0])
    f['volume_pressure'] = (f.get('vol_buy_pct') or 0) - (f.get('vol_sell_pct') or 0)
    f['pattern_net'] = (f.get('pattern_bull_w') or 0) - (f.get('pattern_bear_w') or 0)
    f['mom_combo'] = ((f.get('mom5') or 0) * 0.65) + ((f.get('mom10') or 0) * 0.35)
    f['rsi_centered'] = (f.get('wilder_rsi') or 50) - 50
    f['stoch_centered'] = (f.get('stoch_k') or 50) - 50
    return f

def predict(symbol, features_dict):
    model, meta = load_model(symbol)
    if model is None:
        return {'error': 'model_not_found', 'signal': None, 'confidence': 0,
                'probs': {'UP': 0.33, 'DOWN': 0.33, 'SIDEWAYS': 0.34}}

    if str(meta.get('version', '')).startswith('insufficient'):
        return {'error': 'insufficient_data', 'n_samples': meta.get('n_samples', 0),
                'signal': None, 'confidence': 0,
                'probs': {'UP': 0.33, 'DOWN': 0.33, 'SIDEWAYS': 0.34}}

    feature_cols = meta['feature_cols']
    classes = meta['classes']  # e.g. ['DOWN','SIDEWAYS','UP']

    features_dict = add_derived_features(features_dict)
    X = np.array([[features_dict.get(c, 0) or 0 for c in feature_cols]], dtype=np.float32)
    # Replace NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    probs_raw = model.predict_proba(X)[0]
    pred_idx  = int(np.argmax(probs_raw))
    signal    = classes[pred_idx]
    raw_confidence = float(probs_raw[pred_idx]) * 100
    wf_acc = float(meta.get('wf_avg_balanced_acc') or meta.get('wf_avg_test_acc') or 0)
    quality_factor = min(1.0, max(0.55, wf_acc / 0.55)) if wf_acc else 0.70
    confidence = int(round(50 + (raw_confidence - 50) * quality_factor))

    probs_dict = {c: round(float(p)*100, 1) for c, p in zip(classes, probs_raw)}

    return {
        'signal': signal,
        'confidence': confidence,
        'raw_confidence': int(round(raw_confidence)),
        'probs': probs_dict,
        'model_version': meta.get('version','v1'),
        'wf_acc': meta.get('wf_avg_test_acc', 0),
        'wf_balanced_acc': meta.get('wf_avg_balanced_acc', 0),
        'quality_factor': round(float(quality_factor), 3),
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
