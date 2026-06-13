"""
XGBoost trainer with walk-forward validation.
Usage: python ml-train.py BTC
       python ml-train.py ETH
"""
import sys, json, sqlite3, os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, balanced_accuracy_score
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

DERIVED_FEATURE_COLS = [
    'ema_spread_pct', 'macd_hist_atr', 'bb_width_atr',
    'trend_strength', 'ob_imb_avg', 'volume_pressure',
    'pattern_net', 'mom_combo', 'rsi_centered', 'stoch_centered',
]

MODEL_FEATURE_COLS = FEATURE_COLS + DERIVED_FEATURE_COLS
SIDEWAYS_PNL_THRESHOLD = 0.0015

def load_data():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("""
        SELECT f.*, p.price, p.signal, p.correct, p.result_price, p.result_pnl
        FROM features f
        JOIN predictions p ON p.feature_id = f.id
        WHERE f.symbol=? AND p.checked=1 AND p.result_pnl IS NOT NULL
        ORDER BY f.timestamp ASC
    """, conn, params=(SYMBOL,))
    conn.close()
    return df

def add_derived_features(df):
    df = df.copy()
    atr_safe = df['atr'].replace(0, np.nan)
    bb_width = (df['bb_upper'] - df['bb_lower']).replace(0, np.nan)

    df['ema_spread_pct'] = (df['ema9'] - df['ema21']) / df['ema21'].replace(0, np.nan) * 100
    df['macd_hist_atr'] = df['macd_hist'] / atr_safe
    df['bb_width_atr'] = bb_width / atr_safe
    df['trend_strength'] = df['adx'] * np.sign(df['di_plus'] - df['di_minus'])
    df['ob_imb_avg'] = df[['ob_imb_l5', 'ob_imb_l10', 'ob_imb_l20']].mean(axis=1)
    df['volume_pressure'] = df['vol_buy_pct'] - df['vol_sell_pct']
    df['pattern_net'] = df['pattern_bull_w'] - df['pattern_bear_w']
    df['mom_combo'] = (df['mom5'] * 0.65) + (df['mom10'] * 0.35)
    df['rsi_centered'] = df['wilder_rsi'] - 50
    df['stoch_centered'] = df['stoch_k'] - 50
    return df

def build_outcome_labels(df):
    labels = []
    for _, row in df.iterrows():
        entry_price = float(row.get('price') or 0)
        result_price = float(row.get('result_price') or 0)
        if entry_price <= 0 or result_price <= 0:
            labels.append('SIDEWAYS')
            continue

        future_return = (result_price - entry_price) / entry_price
        if abs(future_return) < SIDEWAYS_PNL_THRESHOLD:
            labels.append('SIDEWAYS')
        elif future_return > 0:
            labels.append('UP')
        else:
            labels.append('DOWN')
    return np.array(labels)

def build_sample_weights(df):
    pnl = df['result_pnl'].fillna(0).abs().clip(lower=0.0005, upper=0.02)
    weights = 1.0 + (pnl / 0.02 * 2.0)
    return weights.values

def prepare_features(df):
    df = add_derived_features(df)
    X = df[MODEL_FEATURE_COLS].copy()
    # Fill NaN with median
    X = X.fillna(X.median())
    le = LabelEncoder()
    y_raw = build_outcome_labels(df)
    class_order = [label for label in ['DOWN', 'SIDEWAYS', 'UP'] if label in set(y_raw)]
    le.fit(class_order)
    y = le.transform(y_raw)
    sample_weight = build_sample_weights(df)
    return X.values, y, le, sample_weight, y_raw

def walk_forward_validate(X, y, sample_weight, n_splits=5):
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
            n_estimators=120, max_depth=3, learning_rate=0.06,
            subsample=0.75, colsample_bytree=0.75,
            min_child_weight=4, gamma=0.15, reg_alpha=0.2, reg_lambda=1.4,
            use_label_encoder=False, eval_metric='mlogloss',
            random_state=42, verbosity=0
        )
        model.fit(X_train, y_train, sample_weight=sample_weight[:train_end])
        train_acc = accuracy_score(y_train, model.predict(X_train))
        test_acc  = accuracy_score(y_test,  model.predict(X_test))
        test_bal_acc = balanced_accuracy_score(y_test, model.predict(X_test))
        results.append({'train_acc': train_acc, 'test_acc': test_acc, 'test_bal_acc': test_bal_acc,
                        'train_size': train_end, 'test_size': test_end - train_end})
        print(f"  Window {i+1}: train={train_acc:.3f} test={test_acc:.3f} "
              f"balanced={test_bal_acc:.3f} (train_n={train_end}, test_n={test_end-train_end})")
    return results

def train_final(X, y, sample_weight):
    """Train final model on all data with conservative regularization"""
    model = XGBClassifier(
        n_estimators=240, max_depth=3, learning_rate=0.04,
        subsample=0.75, colsample_bytree=0.75, min_child_weight=4,
        gamma=0.15, reg_alpha=0.2, reg_lambda=1.4,
        use_label_encoder=False, eval_metric='mlogloss',
        random_state=42, verbosity=0
    )
    model.fit(X, y, sample_weight=sample_weight)
    model.save_model(MODEL_PATH)
    # Feature importance
    importance = dict(zip(MODEL_FEATURE_COLS, model.feature_importances_))
    top = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\nTop 10 features:")
    for f, v in top:
        print(f"  {f}: {v:.4f}")
    train_acc = accuracy_score(y, model.predict(X))
    return model, train_acc, importance

def save_meta(le, wf_results, train_acc, importance, n_samples, class_counts):
    avg_test = np.mean([r['test_acc'] for r in wf_results]) if wf_results else 0
    avg_bal = np.mean([r['test_bal_acc'] for r in wf_results]) if wf_results else 0
    meta = {
        'symbol': SYMBOL,
        'classes': list(le.classes_),
        'feature_cols': MODEL_FEATURE_COLS,
        'n_samples': n_samples,
        'train_acc': round(train_acc, 4),
        'wf_avg_test_acc': round(avg_test, 4),
        'wf_avg_balanced_acc': round(avg_bal, 4),
        'wf_results': wf_results,
        'class_counts': class_counts,
        'label_policy': {
            'target': 'realized_best_direction',
            'sideways_abs_pnl_below': SIDEWAYS_PNL_THRESHOLD,
        },
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
                'feature_cols': MODEL_FEATURE_COLS, 'classes': ['DOWN','SIDEWAYS','UP']}
        with open(META_PATH, 'w') as f: json.dump(meta, f)
        sys.exit(0)

    X, y, le, sample_weight, y_raw = prepare_features(df)
    unique, counts = np.unique(y_raw, return_counts=True)
    class_counts = {str(k): int(v) for k, v in zip(unique, counts)}
    print(f"Outcome labels: {class_counts}")
    if len(class_counts) < 2:
        print("Not enough outcome diversity to train. Need at least two realized classes.")
        meta = {'symbol': SYMBOL, 'n_samples': n, 'version': 'insufficient_classes',
                'feature_cols': MODEL_FEATURE_COLS, 'classes': list(class_counts.keys()),
                'class_counts': class_counts}
        with open(META_PATH, 'w') as f: json.dump(meta, f)
        sys.exit(0)
    print(f"Features: {X.shape[1]}, Classes: {list(le.classes_)}")
    print(f"\n--- Walk-Forward Validation ({min(5, n//10)} splits) ---")
    wf_results = walk_forward_validate(X, y, sample_weight, n_splits=min(5, max(2, n//10)))
    if wf_results:
        avg = np.mean([r['test_acc'] for r in wf_results])
        print(f"\nAvg out-of-sample accuracy: {avg:.3f}")

    print("\n--- Training Final Model ---")
    model, train_acc, importance = train_final(X, y, sample_weight)
    print(f"Final train accuracy: {train_acc:.3f}")

    save_meta(le, wf_results, train_acc, importance, n, class_counts)
    log_to_db(wf_results, train_acc, n)
    print(f"\nModel saved to {MODEL_PATH}")
    print("Done.")

if __name__ == '__main__':
    main()
