/**
 * learning_engine.js
 * This is what makes the system actually learn.
 *
 * It does 4 things:
 * 1. Stores every signal + outcome in SQLite
 * 2. Computes per-indicator accuracy from history
 * 3. Adjusts weights based on what's actually working
 * 4. Triggers XGBoost retraining when enough data exists
 *
 * The system was NOT learning before because:
 * - localStorage resets between sessions
 * - Weights were only nudged ±5% with no memory
 * - Per-indicator accuracy was never persisted
 * - No feedback to the feature builder
 */

const { db, stmts } = require('./database');
const { spawn }     = require('child_process');
const path          = require('path');
const { config }    = require('./config');

// ── Minimum samples before XGBoost trains ────────────────
const MIN_XGB_SAMPLES   = 30;   // train first model
const RETRAIN_INTERVAL  = 25;   // retrain every 25 new results

// ── Weight bounds ─────────────────────────────────────────
const W_MIN = 0.3;
const W_MAX = 5.0;

// ── Load current weights from DB ─────────────────────────
function loadWeights(coin) {
  const row = db.prepare(`
    SELECT weights FROM learned_weights WHERE coin=? ORDER BY updated_at DESC LIMIT 1
  `).get(coin);
  if (row) {
    try { return JSON.parse(row.weights); } catch {}
  }
  return getDefaultWeights();
}

// ── Save weights to DB ────────────────────────────────────
function saveWeights(coin, weights) {
  db.prepare(`
    INSERT OR REPLACE INTO learned_weights (coin, weights, updated_at)
    VALUES (?, ?, ?)
  `).run(coin, JSON.stringify(weights), Date.now());
}

// ── Default weights ───────────────────────────────────────
function getDefaultWeights() {
  return {
    rsi: 2, wilderRsi: 2.5, ema: 2, emaCross: 2.5,
    macd: 1.5, macdCross: 2, bb: 1.5, adx: 1.5,
    vwap: 1.2, vdelta: 1, ob: 1.5, mtf: 1.5,
    mom: 1, pattern: 2.5, spike: 1.5, stochRsi: 2,
    williamsR: 1.5, cci: 1.5, regime: 2, tod: 1,
    cvd: 1.8, liq: 2, orderFlow: 2, lstm: 2.5, rl: 2,
  };
}

// ── Record a completed prediction+outcome ────────────────
function recordOutcome(coin, predId, signal, indSignals, entryPrice, exitPrice, correct, pnlPct, regime) {
  const diff = exitPrice - entryPrice;

  // 1. Update the prediction row
  try {
    stmts.updatePredictionResult.run({
      id:           predId,
      check_time:   Date.now(),
      result_price: exitPrice,
      result_pnl:   pnlPct,
      correct:      correct ? 1 : 0,
    });
  } catch(e) { console.error('[Learning] updatePredictionResult error:', e.message); }

  // 2. Record per-indicator outcomes
  if (indSignals && typeof indSignals === 'object') {
    const stmt = db.prepare(`
      INSERT INTO indicator_outcomes (coin, indicator, signal, correct, pnl_pct, regime, timestamp)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `);
    for (const [ind, sig] of Object.entries(indSignals)) {
      const indCorrect = (sig === 'UP' && diff > 0) || (sig === 'DOWN' && diff < 0) ? 1 : 0;
      stmt.run(coin, ind, sig, indCorrect, pnlPct, regime, Date.now());
    }
  }

  console.log(`[Learning] ${coin} pred#${predId} ${signal} → ${correct?'✓':'✗'} PnL: ${(pnlPct*100).toFixed(3)}%`);

  // 3. Recompute weights
  recomputeWeights(coin);

  // 4. Trigger XGBoost retrain if needed
  checkAndRetrain(coin);
}

// ── Compute per-indicator accuracy and adjust weights ────
function recomputeWeights(coin) {
  // Get accuracy per indicator (last 50 outcomes)
  const rows = db.prepare(`
    SELECT indicator,
           COUNT(*) as total,
           SUM(correct) as wins,
           AVG(pnl_pct) as avg_pnl
    FROM indicator_outcomes
    WHERE coin=?
    GROUP BY indicator
  `).all(coin);

  if (!rows.length) return;

  const weights = loadWeights(coin);
  const changes = [];

  for (const row of rows) {
    if (row.total < 5) continue; // need at least 5 samples
    const acc   = row.wins / row.total;
    const edge  = row.avg_pnl; // positive = indicator predicts profitable moves
    const ind   = row.indicator;
    if (!(ind in weights)) continue;

    const prev = weights[ind];
    // Boost if accuracy > 60% AND positive edge
    if (acc > 0.60 && edge > 0) {
      weights[ind] = Math.min(W_MAX, weights[ind] * (1 + (acc - 0.5) * 0.15));
    }
    // Reduce if accuracy < 45% OR negative edge
    else if (acc < 0.45 || edge < 0) {
      weights[ind] = Math.max(W_MIN, weights[ind] * (1 - (0.5 - acc) * 0.15));
    }
    // Small boost for accuracy 50-60%
    else if (acc >= 0.50) {
      weights[ind] = Math.min(W_MAX, weights[ind] * 1.01);
    }

    const change = weights[ind] - prev;
    if (Math.abs(change) > 0.01) {
      changes.push(`${ind}: ${prev.toFixed(2)} → ${weights[ind].toFixed(2)} (acc:${(acc*100).toFixed(0)}% n:${row.total})`);
    }
  }

  // Global accuracy from predictions table (uses symbol column)
  const globalRow = db.prepare(`
    SELECT COUNT(*) as total, SUM(correct) as wins
    FROM predictions WHERE symbol=? AND checked=1
  `).get(coin);

  if (globalRow && globalRow.total >= 10) {
    const globalAcc = globalRow.wins / globalRow.total;
    if (globalAcc < 0.40) {
      // Overall very poor — reduce all weights slightly (prevent overconfidence)
      Object.keys(weights).forEach(k => { weights[k] = Math.max(W_MIN, weights[k] * 0.97); });
      console.log(`[Learning] ${coin} global acc low (${(globalAcc*100).toFixed(0)}%) — reducing all weights`);
    } else if (globalAcc > 0.65) {
      // Overall strong — small boost
      Object.keys(weights).forEach(k => { weights[k] = Math.min(W_MAX, weights[k] * 1.01); });
    }
  }

  saveWeights(coin, weights);
  if (changes.length) {
    console.log(`[Learning] ${coin} weight updates:\n  ${changes.join('\n  ')}`);
  }

  return weights;
}

// ── Walk-forward validation on stored data ───────────────
function runWalkForward(coin) {
  const rows = db.prepare(`
    SELECT f.*, p.signal, p.correct, p.result_pnl
    FROM features f
    JOIN predictions p ON p.feature_id = f.id
    WHERE f.symbol=? AND p.checked=1
    ORDER BY f.timestamp ASC
  `).all(coin);

  if (rows.length < 30) return null;

  const n      = rows.length;
  const splits = Math.min(5, Math.floor(n / 10));
  const step   = Math.floor(n / (splits + 1));
  const results = [];

  for (let i = 0; i < splits; i++) {
    const trainEnd = step + i * step;
    const testEnd  = Math.min(n, trainEnd + step);
    if (testEnd - trainEnd < 5) continue;

    const testRows = rows.slice(trainEnd, testEnd);
    const correct  = testRows.filter(r => r.correct === 1).length;
    const acc      = correct / testRows.length;

    // Compute PnL on test window
    const pnlSum = testRows.reduce((a, b) => a + (b.result_pnl || 0), 0);

    results.push({
      window:   i + 1,
      trainN:   trainEnd,
      testN:    testEnd - trainEnd,
      accuracy: (acc * 100).toFixed(1),
      avgPnl:   (pnlSum / testRows.length * 100).toFixed(3),
    });

    // Store in DB
    stmts.insertWFResult.run({
      symbol:       coin,
      timestamp:    Date.now(),
      window_start: trainEnd,
      window_end:   testEnd,
      train_acc:    0,
      test_acc:     acc,
      model_version: `wf_${i+1}`,
    });
  }

  console.log(`[WalkForward] ${coin}:`, results);
  return results;
}

// ── Check if XGBoost retraining is needed ────────────────
function checkAndRetrain(coin) {
  const row = stmts.countTrainingData.get({ symbol: coin });
  const n   = row.count;

  if (n < MIN_XGB_SAMPLES) {
    console.log(`[Learning] ${coin} needs ${MIN_XGB_SAMPLES - n} more samples before first XGB train`);
    return;
  }

  // Retrain every RETRAIN_INTERVAL new results
  if (n % RETRAIN_INTERVAL !== 0) return;

  console.log(`[Learning] ${coin} — ${n} samples, triggering XGBoost retrain...`);
  trainXGBoost(coin);
}

// ── Spawn XGBoost training ────────────────────────────────
function summarizeRows(rows) {
  const total = rows.length;
  if (!total) return { total: 0, wins: 0, winRate: null, avgPnl: null };
  const wins = rows.filter(r => r.correct === 1).length;
  const avgPnl = rows.reduce((sum, row) => sum + (row.result_pnl || 0), 0) / total;
  return { total, wins, winRate: wins / total, avgPnl };
}

function getRecentOutcomeRows(coin, limit) {
  return db.prepare(`
    SELECT p.*, f.regime
    FROM predictions p
    LEFT JOIN features f ON p.feature_id = f.id
    WHERE p.symbol=? AND p.checked=1 AND p.result_pnl IS NOT NULL
    ORDER BY p.check_time DESC, p.timestamp DESC
    LIMIT ?
  `).all(coin, limit);
}

function scoreAdaptiveStats(stats, cfg) {
  if (!stats.total || stats.total < cfg.minSamples) {
    return { adjustment: 0, reason: `need ${cfg.minSamples - stats.total} more samples`, active: false };
  }

  const winGap = cfg.targetWinRate - stats.winRate;
  const pnlGap = cfg.minAvgPnlPct - stats.avgPnl;
  let adjustment = 0;
  const reasons = [];

  if (winGap > 0) {
    const winPenalty = Math.min(cfg.maxPenalty, Math.ceil(winGap * 40));
    adjustment += winPenalty;
    reasons.push(`win ${(stats.winRate * 100).toFixed(1)}% < target ${(cfg.targetWinRate * 100).toFixed(1)}%`);
  } else if (winGap < -0.08 && stats.avgPnl > cfg.minAvgPnlPct) {
    const bonus = Math.min(cfg.maxBonus, Math.floor(Math.abs(winGap) * 20));
    adjustment -= bonus;
    if (bonus > 0) reasons.push(`strong win rate ${(stats.winRate * 100).toFixed(1)}%`);
  }

  if (pnlGap > 0) {
    const pnlPenalty = Math.min(cfg.maxPenalty, Math.ceil((pnlGap / Math.max(cfg.minAvgPnlPct, 0.0001)) * 3));
    adjustment += pnlPenalty;
    reasons.push(`avg pnl ${(stats.avgPnl * 100).toFixed(3)}% below target`);
  }

  adjustment = Math.max(-cfg.maxBonus, Math.min(cfg.maxPenalty, adjustment));
  return { adjustment, reason: reasons.join('; ') || 'recent edge acceptable', active: true };
}

function getAdaptiveConfidence(coin, regime, modelVersion) {
  const cfg = config.risk.adaptiveConfidence;
  if (!cfg || !cfg.enabled) {
    return { enabled: false, adjustment: 0, minConfidence: config.risk.minConfidence };
  }

  const rows = getRecentOutcomeRows(coin, cfg.lookback);
  const globalStats = summarizeRows(rows);
  const regimeRows = regime ? rows.filter(r => r.regime === regime) : [];
  const modelRows = modelVersion ? rows.filter(r => r.model_version === modelVersion) : [];

  const globalScore = scoreAdaptiveStats(globalStats, cfg);
  const regimeStats = summarizeRows(regimeRows);
  const regimeScore = regime ? scoreAdaptiveStats(regimeStats, cfg) : { adjustment: 0, reason: 'no regime selected', active: false };
  const modelStats = summarizeRows(modelRows);
  const modelScore = modelVersion ? scoreAdaptiveStats(modelStats, cfg) : { adjustment: 0, reason: 'no model selected', active: false };
  const adjustment = Math.max(globalScore.adjustment, regimeScore.adjustment, modelScore.adjustment);

  return {
    enabled: true,
    adjustment,
    minConfidence: Math.max(1, Math.min(99, config.risk.minConfidence + adjustment)),
    global: { ...globalStats, ...globalScore },
    regime: { name: regime, ...regimeStats, ...regimeScore },
    model: { name: modelVersion, ...modelStats, ...modelScore },
  };
}

function trainXGBoost(coin) {
  return new Promise((resolve) => {
    const pythonBin = process.env.PYTHON_BIN || 'python';
    const proc = spawn(pythonBin, ['ml-train.py', coin], { cwd: path.join(__dirname) });
    let out = '';
    proc.stdout.on('data', d => { out += d.toString(); process.stdout.write(`[XGB-${coin}] ${d}`); });
    proc.stderr.on('data', d => { /* suppress sklearn warnings */ });
    proc.on('error', err => {
      console.error(`[XGB-${coin}] failed to start Python (${err.code || err.message})`);
      resolve(out);
    });
    proc.on('close', code => {
      console.log(`[XGB-${coin}] training done (exit ${code})`);
      resolve(out);
    });
  });
}

// ── Get learning stats for API ────────────────────────────
function getLearningStats(coin) {
  const weights    = loadWeights(coin);
  const globalRow  = db.prepare(`SELECT COUNT(*) as total, SUM(correct) as wins, AVG(result_pnl) as avgPnl FROM predictions WHERE symbol=? AND checked=1`).get(coin) || {};
  const indRows    = db.prepare(`SELECT indicator, COUNT(*) as total, SUM(correct) as wins, AVG(pnl_pct) as avgPnl FROM indicator_outcomes WHERE coin=? GROUP BY indicator ORDER BY wins/COUNT(*) DESC`).all(coin);
  const modelLogs  = stmts.getModelLog.all({ symbol: coin });
  const wfRows     = db.prepare(`SELECT * FROM wf_results WHERE symbol=? ORDER BY timestamp DESC LIMIT 10`).all(coin);
  const adaptiveConfidence = getAdaptiveConfidence(coin, null, null);

  // Regime accuracy — safely skipped if column missing
  let regimeAccRows = [];
  try {
    regimeAccRows = db.prepare(`SELECT regime, COUNT(*) as total, SUM(correct) as wins FROM indicator_outcomes WHERE coin=? AND regime IS NOT NULL AND regime != '' GROUP BY regime`).all(coin);
  } catch(e) { /* column not yet available */ }

  const accuracy = globalRow.total > 0 ? (globalRow.wins / globalRow.total * 100).toFixed(1) : null;

  // Regime-specific accuracy
  const regimeRows = db.prepare(`
    SELECT f.regime, COUNT(*) as total, SUM(p.correct) as wins
    FROM predictions p
    JOIN features f ON p.feature_id = f.id
    WHERE p.symbol=? AND p.checked=1 AND f.regime IS NOT NULL
    GROUP BY f.regime
  `).all(coin);

  return {
    coin,
    totalPredictions: globalRow.total || 0,
    accuracy,
    avgPnl: globalRow.avgPnl ? (globalRow.avgPnl * 100).toFixed(4) + '%' : null,
    weights,
    indicatorAccuracy: indRows.map(r => ({
      indicator: r.indicator,
      accuracy:  r.total > 0 ? (r.wins / r.total * 100).toFixed(1) : '—',
      total:     r.total,
      avgPnl:    (r.avgPnl * 100).toFixed(4) + '%',
      weight:    (weights[r.indicator] || 0).toFixed(2),
    })),
    regimeAccuracy: regimeRows.map(r => ({
      regime:   r.regime,
      accuracy: (r.wins / r.total * 100).toFixed(1),
      total:    r.total,
    })),
    modelLog:  modelLogs,
    wfResults: wfRows,
    adaptiveConfidence,
    nextTrainAt: MIN_XGB_SAMPLES + Math.ceil(((globalRow.total || 0) - MIN_XGB_SAMPLES) / RETRAIN_INTERVAL + 1) * RETRAIN_INTERVAL,
  };
}

// ── Ensure DB tables exist ────────────────────────────────
db.exec(`
  CREATE TABLE IF NOT EXISTS learned_weights (
    coin       TEXT PRIMARY KEY,
    weights    TEXT NOT NULL,
    updated_at INTEGER NOT NULL
  );

  CREATE TABLE IF NOT EXISTS indicator_outcomes (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    coin      TEXT NOT NULL,
    indicator TEXT NOT NULL,
    signal    TEXT NOT NULL,
    correct   INTEGER NOT NULL,
    pnl_pct   REAL,
    regime    TEXT,
    timestamp INTEGER NOT NULL
  );
`);

// Safely add regime column if it doesn't exist yet (for existing databases)
try {
  db.exec(`ALTER TABLE indicator_outcomes ADD COLUMN regime TEXT`);
  console.log('[DB] Added regime column to indicator_outcomes');
} catch(e) {
  // Column already exists — ignore
}

module.exports = {
  loadWeights,
  saveWeights,
  getDefaultWeights,
  recordOutcome,
  recomputeWeights,
  runWalkForward,
  trainXGBoost,
  checkAndRetrain,
  getLearningStats,
  getAdaptiveConfidence,
};
