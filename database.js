const Database = require('better-sqlite3');
const path = require('path');

const db = new Database(path.join(__dirname, 'crypto_predictor.db'));

// Enable WAL mode for better concurrent performance
db.pragma('journal_mode = WAL');
db.pragma('synchronous = NORMAL');

// ── Schema ────────────────────────────────────────────────

db.exec(`
  -- Historical candle data
  CREATE TABLE IF NOT EXISTS candles (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT NOT NULL,
    interval  TEXT NOT NULL,
    open_time INTEGER NOT NULL,
    open      REAL, high REAL, low REAL, close REAL, volume REAL,
    UNIQUE(symbol, interval, open_time)
  );

  -- Feature snapshots at prediction time
  CREATE TABLE IF NOT EXISTS features (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    timestamp   INTEGER NOT NULL,
    rsi         REAL, wilder_rsi REAL, stoch_k REAL, stoch_d REAL,
    williams_r  REAL, cci REAL,
    ema9        REAL, ema21 REAL, ema50 REAL,
    macd        REAL, macd_signal REAL, macd_hist REAL,
    bb_upper    REAL, bb_middle REAL, bb_lower REAL, bb_bw REAL,
    atr         REAL, adx REAL, di_plus REAL, di_minus REAL,
    vwap        REAL, cvd REAL,
    ob_imb_l5   REAL, ob_imb_l10 REAL, ob_imb_l20 REAL,
    vol_buy_pct REAL, vol_sell_pct REAL,
    regime      TEXT, tod_session TEXT, tod_score REAL,
    mom5        REAL, mom10 REAL,
    pattern_bull_w REAL, pattern_bear_w REAL,
    vol_spike_ratio REAL
  );

  -- Predictions
  CREATE TABLE IF NOT EXISTS predictions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol       TEXT NOT NULL,
    timestamp    INTEGER NOT NULL,
    price        REAL NOT NULL,
    signal       TEXT NOT NULL,
    confidence   INTEGER NOT NULL,
    prob_up      REAL, prob_down REAL, prob_side REAL,
    ensemble_mom REAL, ensemble_mr REAL, ensemble_vol REAL,
    model_version TEXT DEFAULT 'v1',
    feature_id   INTEGER REFERENCES features(id),
    checked      INTEGER DEFAULT 0,
    check_time   INTEGER,
    result_price REAL,
    result_pnl   REAL,
    correct      INTEGER
  );

  -- Walk-forward validation results
  CREATE TABLE IF NOT EXISTS wf_results (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    timestamp   INTEGER NOT NULL,
    window_start INTEGER, window_end INTEGER,
    train_acc   REAL, test_acc REAL,
    model_version TEXT
  );

  -- Model versions + accuracy log
  CREATE TABLE IF NOT EXISTS model_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT NOT NULL,
    timestamp   INTEGER NOT NULL,
    version     TEXT NOT NULL,
    train_acc   REAL, val_acc REAL,
    n_samples   INTEGER,
    notes       TEXT
  );

  -- Regime history
  CREATE TABLE IF NOT EXISTS regime_history (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol    TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    regime    TEXT NOT NULL,
    adx       REAL, volatility REAL,
    confidence REAL
  );
  CREATE TABLE IF NOT EXISTS learned_weights (
  coin TEXT PRIMARY KEY,
  weights TEXT NOT NULL,
  updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS indicator_outcomes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  coin TEXT NOT NULL,
  indicator TEXT NOT NULL,
  signal TEXT NOT NULL,
  correct INTEGER NOT NULL,
  pnl_pct REAL,
  regime TEXT,
  timestamp INTEGER NOT NULL
);
`);

// ── Prepared statements ───────────────────────────────────

const stmts = {
  insertCandle: db.prepare(`
    INSERT OR REPLACE INTO candles (symbol, interval, open_time, open, high, low, close, volume)
    VALUES (@symbol, @interval, @open_time, @open, @high, @low, @close, @volume)
  `),

  insertFeatures: db.prepare(`
    INSERT INTO features (symbol, timestamp, rsi, wilder_rsi, stoch_k, stoch_d,
      williams_r, cci, ema9, ema21, ema50, macd, macd_signal, macd_hist,
      bb_upper, bb_middle, bb_lower, bb_bw, atr, adx, di_plus, di_minus,
      vwap, cvd, ob_imb_l5, ob_imb_l10, ob_imb_l20, vol_buy_pct, vol_sell_pct,
      regime, tod_session, tod_score, mom5, mom10,
      pattern_bull_w, pattern_bear_w, vol_spike_ratio)
    VALUES (@symbol, @timestamp, @rsi, @wilder_rsi, @stoch_k, @stoch_d,
      @williams_r, @cci, @ema9, @ema21, @ema50, @macd, @macd_signal, @macd_hist,
      @bb_upper, @bb_middle, @bb_lower, @bb_bw, @atr, @adx, @di_plus, @di_minus,
      @vwap, @cvd, @ob_imb_l5, @ob_imb_l10, @ob_imb_l20, @vol_buy_pct, @vol_sell_pct,
      @regime, @tod_session, @tod_score, @mom5, @mom10,
      @pattern_bull_w, @pattern_bear_w, @vol_spike_ratio)
  `),

  insertPrediction: db.prepare(`
    INSERT INTO predictions (symbol, timestamp, price, signal, confidence,
      prob_up, prob_down, prob_side, ensemble_mom, ensemble_mr, ensemble_vol,
      model_version, feature_id)
    VALUES (@symbol, @timestamp, @price, @signal, @confidence,
      @prob_up, @prob_down, @prob_side, @ensemble_mom, @ensemble_mr, @ensemble_vol,
      @model_version, @feature_id)
  `),

  updatePredictionResult: db.prepare(`
    UPDATE predictions SET
      checked=1, check_time=@check_time, result_price=@result_price,
      result_pnl=@result_pnl, correct=@correct
    WHERE id=@id
  `),

  getPendingPredictions: db.prepare(`
    SELECT * FROM predictions
    WHERE checked=0 AND timestamp < @cutoff
    ORDER BY timestamp ASC
  `),

  getRecentPredictions: db.prepare(`
    SELECT * FROM predictions
    WHERE symbol=@symbol
    ORDER BY timestamp DESC LIMIT @limit
  `),

  getTrainingData: db.prepare(`
    SELECT f.*, p.signal, p.correct, p.result_pnl
    FROM features f
    JOIN predictions p ON p.feature_id = f.id
    WHERE f.symbol=@symbol AND p.checked=1
    ORDER BY f.timestamp ASC
  `),

  getCandles: db.prepare(`
    SELECT * FROM candles
    WHERE symbol=@symbol AND interval=@interval
    ORDER BY open_time DESC LIMIT @limit
  `),

  countTrainingData: db.prepare(`
    SELECT COUNT(*) as count FROM predictions
    WHERE symbol=@symbol AND checked=1
  `),

  insertWFResult: db.prepare(`
    INSERT INTO wf_results (symbol, timestamp, window_start, window_end, train_acc, test_acc, model_version)
    VALUES (@symbol, @timestamp, @window_start, @window_end, @train_acc, @test_acc, @model_version)
  `),

  insertModelLog: db.prepare(`
    INSERT INTO model_log (symbol, timestamp, version, train_acc, val_acc, n_samples, notes)
    VALUES (@symbol, @timestamp, @version, @train_acc, @val_acc, @n_samples, @notes)
  `),

  getModelLog: db.prepare(`
    SELECT * FROM model_log WHERE symbol=@symbol ORDER BY timestamp DESC LIMIT 10
  `),

  insertRegime: db.prepare(`
    INSERT INTO regime_history (symbol, timestamp, regime, adx, volatility, confidence)
    VALUES (@symbol, @timestamp, @regime, @adx, @volatility, @confidence)
  `),

  getStats: db.prepare(`
    SELECT
      COUNT(*) as total,
      SUM(correct) as wins,
      AVG(result_pnl) as avg_pnl,
      MIN(result_pnl) as worst,
      MAX(result_pnl) as best
    FROM predictions
    WHERE symbol=@symbol AND checked=1
  `)
};

// ── Batch candle insert ───────────────────────────────────
const insertCandlesBatch = db.transaction((candles) => {
  for (const c of candles) stmts.insertCandle.run(c);
});

module.exports = { db, stmts, insertCandlesBatch };
