"""
XGBoost trainer with walk-forward validation.
Usage: python ml_train.py BTC
       python ml_train.py ETH
"""
import sys, json, sqlite3, os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

SYMBOL = sys.argv[1].upper() if len(sys.argv) > 1 else 'BTC'
DB_PATH = os.path.join(os.path.dirname(__file__), 'crypto_predictor.db')
MODEL_PATH = os.path.join(os.path.dirname(__file__), f'model_{SYMBOL}.json')
META_PATH  = os.path.join(os.path.dirname(__file__), f'model_{SYMBOL}_meta.json')

FEATURE_COLS = [
    'rsi','wilder_rsi','stoch_k','stoch_d','williams_r','cci',
    'ema9','ema21','ema50','macd','macd_signal','macd_hist',
    'bb_upper','bb_middle','bb_lower','bb_bw',
    'atr','adx','di_plus','di_minus',
    'vwap','cvd','ob_imb_l5','ob_imb_l10','ob_imb_l20',
    'vol_buy_pct','vol_sell_pct','tod_score',
    'mom5','mom10','pattern_bull_w','pattern_bear_w','vol_spike_ratio'
]

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(f"""
        SELECT f.*, p.signal, p.correct, p.result_pnl
        FROM features f
        JOIN predictions p ON p.feature_id = f.id
        WHERE f.symbol='{SYMBOL}' AND p.checked=1
        ORDER BY f.timestamp ASC
    """, conn)
    conn.close()
    return df

def prepare_features(df):
    X = df[FEATURE_COLS].copy()
    # Fill NaN with median
    X = X.fillna(X.median())
    # Encode regime as numeric via correlation-friendly mapping
    le = LabelEncoder()
    y_raw = df['signal'].values
    le.fit(['UP','DOWN','SIDEWAYS'])
    y = le.transform(y_raw)
    return X.values, y, le

def walk_forward_validate(X, y, n_splits=5):
    """Time-series walk-forward cross validation"""
    n = len(X)
    min_train = max(30, n // (n_splits + 1))
    step = (n - min_train) // n_splits
    results = []
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end  = min(train_end + step, n)
        if test_end <= train_end: break
        X_train, y_train = X[:train_end], y[:train_end]
        X_test,  y_test  = X[train_end:test_end], y[train_end:test_end]
        model = XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, verbosity=0
        )
        model.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        results.append({'train_acc': train_acc, 'test_acc': test_acc,
                        'train_size': train_end, 'test_size': test_end - train_end})
        print(f"  Window {i+1}: train={train_acc:.3f} test={test_acc:.3f} "
              f"(train_n={train_end}, test_n={test_end-train_end})")
    return results

def train_final(X, y, le):
    """Train final model on all data with calibration"""
    model = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, verbosity=0
    )
    # Calibrate probabilities using isotonic regression
    calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calibrated.fit(X, y)
    # Save raw booster for fast inference
    model.fit(X, y)
    model.save_model(MODEL_PATH)
    # Feature importance
    importance = dict(zip(FEATURE_COLS, model.feature_importances_))
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 features:")
    for f, v in top:
        print(f"  {f}: {v:.4f}")
    train_acc = accuracy_score(y, model.predict(X))
    return model, calibrated, train_acc, importance

def save_meta(le, wf_results, train_acc, importance, n_samples):
    avg_test = np.mean([r['test_acc'] for r in wf_results]) if wf_results else 0
    meta = {
        'symbol': SYMBOL,
        'classes': list(le.classes_),
        'feature_cols': FEATURE_COLS,
        'n_samples': n_samples,
        'train_acc': round(train_acc, 4),
        'wf_avg_test_acc': round(avg_test, 4),
        'wf_results': wf_results,
        'top_features': {k: round(float(v), 4) for k, v in
                         sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]},
        'version': f'v{n_samples}'
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"\nMeta saved to {META_PATH}")

def log_to_db(wf_results, train_acc, n_samples):
    avg_test = np.mean([r['test_acc'] for r in wf_results]) if wf_results else 0
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        INSERT INTO model_log (symbol, timestamp, version, train_acc, val_acc, n_samples, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (SYMBOL, int(pd.Timestamp.now().timestamp()*1000),
          f'v{n_samples}', round(train_acc,4), round(avg_test,4),
          n_samples, f'WF splits={len(wf_results)}'))
    conn.commit()
    conn.close()

def main():
    print(f"=== XGBoost Training — {SYMBOL} ===")
    df = load_data()
    n = len(df)
    print(f"Training samples: {n}")
    if n < 20:
        print(f"Not enough data (need 20+, have {n}). Run the predictor longer to collect data.")
        # Save dummy meta so server knows model needs more data
        meta = {'symbol': SYMBOL, 'n_samples': n, 'version': 'insufficient',
                'feature_cols': FEATURE_COLS, 'classes': ['DOWN','SIDEWAYS','UP']}
        with open(META_PATH, 'w') as f: json.dump(meta, f)
        sys.exit(0)

    X, y, le = prepare_features(df)
    print(f"Features: {X.shape[1]}, Classes: {list(le.classes_)}")
    print(f"\n--- Walk-Forward Validation ({min(5, n//10)} splits) ---")
    wf_results = walk_forward_validate(X, y, n_splits=min(5, max(2, n//10)))
    if wf_results:
        avg = np.mean([r['test_acc'] for r in wf_results])
        print(f"\nAvg out-of-sample accuracy: {avg:.3f}")

    print("\n--- Training Final Model ---")
    model, calibrated, train_acc, importance = train_final(X, y, le)
    print(f"Final train accuracy: {train_acc:.3f}")

    save_meta(le, wf_results, train_acc, importance, n)
    log_to_db(wf_results, train_acc, n)
    print(f"\nModel saved to {MODEL_PATH}")
    print("Done.")

if __name__ == '__main__':
    main()
