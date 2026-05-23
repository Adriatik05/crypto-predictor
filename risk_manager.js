/**
 * risk_manager.js
 * Hard risk rules — called before every trade decision
 * Enforces: stop loss, take profit, daily loss limit,
 * max positions, confidence threshold, slippage model
 */

const { stmts, db } = require('./database');

// ── Default risk config ───────────────────────────────────
const DEFAULT_CONFIG = {
  // Per-trade
  stopLossPct:       0.005,   // 0.5% hard stop
  takeProfitPct:     0.010,   // 1.0% take profit
  maxPositionPct:    0.10,    // max 10% of capital per trade (Kelly will size down further)
  feePct:            0.001,   // 0.1% Binance maker fee per side
  slippagePct:       0.0005,  // 0.05% estimated slippage

  // Session limits
  dailyLossLimitPct: 0.02,    // stop trading after -2% on capital in one day
  maxOpenPositions:  1,       // only 1 trade open at a time (conservative)
  maxTradesPerDay:   20,      // cap daily trade count

  // Signal quality gates
  minConfidence:     72,      // minimum ML confidence to enter
  minAgreement:      0.80,    // minimum model agreement (4/5 models)
  minRegimes:        ['bull','bear'], // only trade in trending regimes
  requireMACDCross:  false,   // optional: require MACD crossover confirmation

  // Minimum edge after fees
  minExpectedEdgePct: 0.003,  // signal must show >0.3% expected edge to enter
};

// ── In-memory session state (reset daily) ────────────────
const session = {
  BTC: { dailyPnl: 0, tradesCount: 0, openPositions: 0, lastTradeTime: 0, dailyReset: '' },
  ETH: { dailyPnl: 0, tradesCount: 0, openPositions: 0, lastTradeTime: 0, dailyReset: '' },
};

function getToday() {
  return new Date().toISOString().slice(0, 10);
}

function resetDailyIfNeeded(coin) {
  const today = getToday();
  if (session[coin].dailyReset !== today) {
    session[coin].dailyPnl    = 0;
    session[coin].tradesCount = 0;
    session[coin].dailyReset  = today;
    console.log(`[Risk] ${coin} daily session reset for ${today}`);
  }
}

// ── Main gate: should we trade this signal? ──────────────
function shouldTrade(coin, signal, confidence, agreement, regime, price, capital, config = DEFAULT_CONFIG) {
  resetDailyIfNeeded(coin);
  const s = session[coin];
  const reasons = [];

  // 1. Confidence gate
  if (confidence < config.minConfidence) {
    reasons.push(`conf ${confidence} < min ${config.minConfidence}`);
  }

  // 2. Model agreement gate
  if (agreement < config.minAgreement) {
    reasons.push(`agreement ${(agreement*100).toFixed(0)}% < min ${config.minAgreement*100}%`);
  }

  // 3. Regime gate
  if (config.minRegimes.length > 0 && !config.minRegimes.includes(regime)) {
    reasons.push(`regime '${regime}' not in allowed list`);
  }

  // 4. Daily loss limit
  if (s.dailyPnl <= -config.dailyLossLimitPct) {
    reasons.push(`daily loss limit hit (${(s.dailyPnl*100).toFixed(2)}%)`);
  }

  // 5. Max trades per day
  if (s.tradesCount >= config.maxTradesPerDay) {
    reasons.push(`max ${config.maxTradesPerDay} trades/day reached`);
  }

  // 6. Max open positions
  if (s.openPositions >= config.maxOpenPositions) {
    reasons.push(`max ${config.maxOpenPositions} open positions`);
  }

  // 7. Minimum edge after fees
  const roundTripCost = (config.feePct + config.slippagePct) * 2;
  const expectedEdge  = (confidence / 100 - 0.5) * config.stopLossPct * 2; // rough edge estimate
  if (expectedEdge < config.minExpectedEdgePct + roundTripCost) {
    reasons.push(`edge ${(expectedEdge*100).toFixed(3)}% < min after fees`);
  }

  const allowed = reasons.length === 0;
  return { allowed, reasons };
}

// ── Kelly position sizing ─────────────────────────────────
function kellySize(capital, winRate, avgWin, avgLoss, config = DEFAULT_CONFIG) {
  if (!winRate || !avgWin || !avgLoss || avgLoss === 0) {
    // No history yet — use conservative fixed size
    return capital * 0.02; // 2% of capital until we have data
  }
  const b  = avgWin / avgLoss;
  const kelly = (winRate * b - (1 - winRate)) / b;
  // Use half-Kelly for safety, cap at maxPositionPct
  const fraction = Math.max(0, Math.min(config.maxPositionPct, kelly * 0.5));
  return capital * fraction;
}

// ── Stop loss and take profit prices ────────────────────
function getExitLevels(signal, entryPrice, config = DEFAULT_CONFIG) {
  if (signal === 'UP') {
    return {
      stopLoss:   entryPrice * (1 - config.stopLossPct),
      takeProfit: entryPrice * (1 + config.takeProfitPct),
    };
  } else {
    return {
      stopLoss:   entryPrice * (1 + config.stopLossPct),
      takeProfit: entryPrice * (1 - config.takeProfitPct),
    };
  }
}

// ── Calculate real entry price after slippage + fees ────
function getRealEntryPrice(signal, marketPrice, config = DEFAULT_CONFIG) {
  const slip = signal === 'UP' ? config.slippagePct : -config.slippagePct;
  return marketPrice * (1 + slip + config.feePct);
}

// ── Record trade open ─────────────────────────────────────
function onTradeOpen(coin) {
  resetDailyIfNeeded(coin);
  session[coin].openPositions++;
  session[coin].tradesCount++;
  session[coin].lastTradeTime = Date.now();
}

// ── Record trade close ────────────────────────────────────
function onTradeClose(coin, pnlPct) {
  session[coin].openPositions = Math.max(0, session[coin].openPositions - 1);
  session[coin].dailyPnl     += pnlPct;
}

// ── Get session summary ───────────────────────────────────
function getSessionSummary(coin) {
  resetDailyIfNeeded(coin);
  const s = session[coin];
  const cfg = DEFAULT_CONFIG;
  return {
    dailyPnl:          s.dailyPnl,
    dailyPnlPct:       (s.dailyPnl * 100).toFixed(2) + '%',
    tradesCount:       s.tradesCount,
    openPositions:     s.openPositions,
    dailyLimitHit:     s.dailyPnl <= -cfg.dailyLossLimitPct,
    tradeLimitHit:     s.tradesCount >= cfg.maxTradesPerDay,
    remainingCapacity: Math.max(0, cfg.maxTradesPerDay - s.tradesCount),
    config:            cfg,
  };
}

// ── Slippage model (tiered by order size) ────────────────
function estimateSlippage(coin, orderSizeUSD) {
  // BTC has deeper liquidity than ETH
  const base = coin === 'BTC' ? 0.0002 : 0.0004;
  if (orderSizeUSD < 1000)   return base;
  if (orderSizeUSD < 10000)  return base * 1.5;
  if (orderSizeUSD < 100000) return base * 3;
  return base * 6; // large order — significant impact
}

module.exports = {
  DEFAULT_CONFIG,
  shouldTrade,
  kellySize,
  getExitLevels,
  getRealEntryPrice,
  onTradeOpen,
  onTradeClose,
  getSessionSummary,
  estimateSlippage,
  resetDailyIfNeeded,
};
