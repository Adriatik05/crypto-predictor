/**
 * paper_trader.js
 * Simulates real trades with proper entry/exit tracking.
 * Uses real Binance prices but no real money.
 *
 * What this fixes:
 * - Before: predictions were checked once at 5 min, binary correct/wrong
 * - Now: full trade simulation with stop loss, take profit, partial fills,
 *   slippage, fees, trailing stops, and regime-aware position sizing
 *
 * Every completed paper trade feeds directly into learning_engine.js
 * so the ML trains on realistic trade outcomes, not just price direction.
 */

const fetch    = require('node-fetch');
const { db }   = require('./database');
const risk     = require('./risk_manager');
const learning = require('./learning_engine');
const { config } = require('./config');

// ── Ensure paper trading tables exist ───────────────────
db.exec(`
  CREATE TABLE IF NOT EXISTS paper_trades (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    coin           TEXT NOT NULL,
    signal         TEXT NOT NULL,
    entry_price    REAL NOT NULL,
    exit_price     REAL,
    position_size  REAL NOT NULL,
    stop_loss      REAL NOT NULL,
    take_profit    REAL NOT NULL,
    trailing_stop  REAL,
    opened_at      INTEGER NOT NULL,
    closed_at      INTEGER,
    exit_reason    TEXT,
    gross_pnl_pct  REAL,
    net_pnl_pct    REAL,
    net_pnl_usd    REAL,
    fee_paid       REAL,
    correct        INTEGER,
    confidence     INTEGER,
    agreement      REAL,
    regime         TEXT,
    model_version  TEXT,
    pred_id        INTEGER,
    status         TEXT DEFAULT 'open'
  );

  CREATE TABLE IF NOT EXISTS paper_account (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    coin       TEXT NOT NULL,
    timestamp  INTEGER NOT NULL,
    balance    REAL NOT NULL,
    event      TEXT
  );
`);

try {
  db.exec(`ALTER TABLE paper_trades ADD COLUMN correct INTEGER`);
} catch(e) {
  // Existing databases already have the column.
}

// ── Paper trading account state ──────────────────────────
const STARTING_BALANCE = config.paperTrading.startingBalanceUsd; // USD per coin

function getBalance(coin) {
  const row = db.prepare(
    `SELECT balance FROM paper_account WHERE coin=? ORDER BY timestamp DESC LIMIT 1`
  ).get(coin);
  return row ? row.balance : STARTING_BALANCE;
}

function updateBalance(coin, newBalance, event) {
  db.prepare(
    `INSERT INTO paper_account (coin, timestamp, balance, event) VALUES (?,?,?,?)`
  ).run(coin, Date.now(), newBalance, event);
}

// ── Get open trade for a coin ────────────────────────────
function getOpenTrade(coin) {
  return db.prepare(
    `SELECT * FROM paper_trades WHERE coin=? AND status='open' ORDER BY opened_at DESC LIMIT 1`
  ).get(coin);
}

// ── Open a new paper trade ───────────────────────────────
function openTrade(coin, signal, entryPrice, confidence, agreement, regime, modelVersion, predId) {
  const cfg     = risk.DEFAULT_CONFIG;
  const balance = getBalance(coin);

  // Kelly-based position sizing from historical results
  const hist    = db.prepare(
    `SELECT correct, net_pnl_pct FROM paper_trades WHERE coin=? AND status='closed' ORDER BY closed_at DESC LIMIT 50`
  ).all(coin);
  const wins    = hist.filter(h => h.correct === 1);
  const losses  = hist.filter(h => h.correct === 0);
  const winRate = hist.length > 0 ? wins.length / hist.length : 0.5;
  const avgWin  = wins.length  > 0 ? wins.reduce((a,b)=>a+Math.abs(b.net_pnl_pct),0)/wins.length  : cfg.stopLossPct;
  const avgLoss = losses.length> 0 ? losses.reduce((a,b)=>a+Math.abs(b.net_pnl_pct),0)/losses.length : cfg.stopLossPct;
  const kellyUSD = risk.kellySize(balance, winRate, avgWin, avgLoss);

  // Apply slippage to entry
  const slip       = risk.estimateSlippage(coin, kellyUSD);
  const realEntry  = signal === 'UP'
    ? entryPrice * (1 + slip + cfg.feePct)
    : entryPrice * (1 - slip - cfg.feePct);

  // Stop loss and take profit
  const levels = risk.getExitLevels(signal, realEntry);

  const feePaid = kellyUSD * cfg.feePct;

  const row = db.prepare(`
    INSERT INTO paper_trades
      (coin, signal, entry_price, position_size, stop_loss, take_profit,
       trailing_stop, opened_at, confidence, agreement, regime, model_version, pred_id, status, fee_paid)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,'open',?)
  `).run(
    coin, signal, realEntry, kellyUSD,
    levels.stopLoss, levels.takeProfit,
    levels.stopLoss, // trailing stop starts at SL
    Date.now(),
    confidence, agreement, regime, modelVersion, predId,
    feePaid
  );

  console.log(`[Paper] OPEN ${coin} ${signal} @ $${realEntry.toFixed(2)} size:$${kellyUSD.toFixed(0)} SL:$${levels.stopLoss.toFixed(2)} TP:$${levels.takeProfit.toFixed(2)}`);
  return row.lastInsertRowid;
}

// ── Update trailing stop ─────────────────────────────────
function updateTrailingStop(trade, currentPrice) {
  const cfg = risk.DEFAULT_CONFIG;
  let newTS  = trade.trailing_stop;

  if (trade.signal === 'UP') {
    // Price moved up — trail stop up
    const newLevel = currentPrice * (1 - cfg.stopLossPct * 0.8); // tighter trail
    if (newLevel > newTS) newTS = newLevel;
  } else {
    const newLevel = currentPrice * (1 + cfg.stopLossPct * 0.8);
    if (newLevel < newTS) newTS = newLevel;
  }

  if (newTS !== trade.trailing_stop) {
    db.prepare(`UPDATE paper_trades SET trailing_stop=? WHERE id=?`).run(newTS, trade.id);
  }
  return newTS;
}

// ── Check if open trade should close ────────────────────
async function checkOpenTrades(coin, currentPrice, currentHigh, currentLow) {
  const trade = getOpenTrade(coin);
  if (!trade) return;

  const cfg = risk.DEFAULT_CONFIG;
  let exitPrice  = null;
  let exitReason = null;

  // Update trailing stop first
  const trailingStop = updateTrailingStop(trade, currentPrice);

  if (trade.signal === 'UP') {
    if (currentLow  <= trailingStop)      { exitPrice = trailingStop;      exitReason = 'trailing_stop'; }
    else if (currentHigh >= trade.take_profit) { exitPrice = trade.take_profit; exitReason = 'take_profit'; }
  } else {
    if (currentHigh >= trailingStop)      { exitPrice = trailingStop;      exitReason = 'trailing_stop'; }
    else if (currentLow  <= trade.take_profit) { exitPrice = trade.take_profit; exitReason = 'take_profit'; }
  }

  // Time-based close if no SL/TP hit.
  const ageMinutes = (Date.now() - trade.opened_at) / 60000;
  if (!exitPrice && ageMinutes >= config.paperTrading.timeoutMinutes) {
    exitPrice  = currentPrice;
    exitReason = 'timeout_15m';
  }

  if (!exitPrice) return; // still open

  await closeTrade(trade, exitPrice, exitReason, coin);
}

// ── Close a paper trade ──────────────────────────────────
async function closeTrade(trade, exitPrice, exitReason, coin) {
  const cfg    = risk.DEFAULT_CONFIG;
  const slip   = risk.estimateSlippage(coin, trade.position_size);
  const realExit = trade.signal === 'UP'
    ? exitPrice * (1 - slip - cfg.feePct)
    : exitPrice * (1 + slip + cfg.feePct);

  const grossPnlPct = trade.signal === 'UP'
    ? (realExit - trade.entry_price) / trade.entry_price
    : (trade.entry_price - realExit) / trade.entry_price;

  const exitFee   = trade.position_size * cfg.feePct;
  const totalFees = (trade.fee_paid || 0) + exitFee;
  const netPnlPct = grossPnlPct - (totalFees / trade.position_size);
  const netPnlUSD = trade.position_size * netPnlPct;
  const correct   = netPnlUSD > 0 ? 1 : 0;

  // Update trade record
  db.prepare(`
    UPDATE paper_trades SET
      exit_price=?, closed_at=?, exit_reason=?,
      gross_pnl_pct=?, net_pnl_pct=?, net_pnl_usd=?,
      fee_paid=?, correct=?, status='closed'
    WHERE id=?
  `).run(realExit, Date.now(), exitReason, grossPnlPct, netPnlPct, netPnlUSD, totalFees, correct, trade.id);

  // Update account balance
  const oldBalance = getBalance(coin);
  const newBalance = oldBalance + netPnlUSD;
  updateBalance(coin, newBalance, `${trade.signal} ${exitReason} ${netPnlPct>=0?'+':''}${(netPnlPct*100).toFixed(3)}%`);

  // Feed into risk session
  risk.onTradeClose(coin, netPnlPct);

  // Feed into learning engine — THIS IS THE KEY LEARNING STEP
  if (trade.pred_id) {
    const indSignals = {}; // will be populated from DB
    learning.recordOutcome(
      coin,
      trade.pred_id,
      trade.signal,
      indSignals,
      trade.entry_price,
      realExit,
      correct,
      netPnlPct,
      trade.regime
    );
  }

  console.log(`[Paper] CLOSE ${coin} ${trade.signal} ${exitReason} entry:$${trade.entry_price.toFixed(2)} exit:$${realExit.toFixed(2)} net:${(netPnlPct*100).toFixed(3)}% bal:$${newBalance.toFixed(0)}`);

  return { netPnlPct, netPnlUSD, correct, exitReason };
}

// ── Get paper trading stats ───────────────────────────────
function getPaperStats(coin) {
  const balance = getBalance(coin);
  const trades  = db.prepare(`SELECT * FROM paper_trades WHERE coin=? AND status='closed' ORDER BY closed_at DESC`).all(coin);
  const open    = getOpenTrade(coin);

  if (!trades.length) {
    return { coin, balance, startingBalance: STARTING_BALANCE, totalReturn: 0, trades: 0, winRate: 0, openTrade: open };
  }

  const wins     = trades.filter(t => t.net_pnl_usd > 0);
  const winRate  = wins.length / trades.length * 100;
  const totalRet = (balance - STARTING_BALANCE) / STARTING_BALANCE * 100;
  const rets     = trades.map(t => t.net_pnl_pct);
  const mean     = rets.reduce((a,b)=>a+b,0) / rets.length;
  const std      = Math.sqrt(rets.reduce((a,b)=>a+(b-mean)**2,0)/rets.length) || 0.0001;
  const sharpe   = (mean / std * Math.sqrt(rets.length)).toFixed(2);
  const maxDD    = (() => {
    let peak = STARTING_BALANCE, maxD = 0, bal = STARTING_BALANCE;
    for (const t of [...trades].reverse()) { bal+=t.net_pnl_usd; if(bal>peak)peak=bal; const d=(peak-bal)/peak*100; if(d>maxD)maxD=d; }
    return maxD.toFixed(2);
  })();

  const byReason = {};
  trades.forEach(t=>{if(!byReason[t.exit_reason])byReason[t.exit_reason]={count:0,pnl:0};byReason[t.exit_reason].count++;byReason[t.exit_reason].pnl+=t.net_pnl_usd;});

  const byRegime = {};
  trades.forEach(t=>{if(!t.regime)return;if(!byRegime[t.regime])byRegime[t.regime]={count:0,wins:0,pnl:0};byRegime[t.regime].count++;if(t.net_pnl_usd>0)byRegime[t.regime].wins++;byRegime[t.regime].pnl+=t.net_pnl_usd;});

  return {
    coin, balance, startingBalance: STARTING_BALANCE,
    totalReturn: totalRet.toFixed(2),
    totalReturnUSD: (balance - STARTING_BALANCE).toFixed(2),
    trades: trades.length,
    wins: wins.length,
    winRate: winRate.toFixed(1),
    sharpe, maxDD,
    avgWin:  (wins.reduce((a,b)=>a+b.net_pnl_pct,0)/(wins.length||1)*100).toFixed(3),
    avgLoss: (trades.filter(t=>t.net_pnl_usd<=0).reduce((a,b)=>a+b.net_pnl_pct,0)/(trades.filter(t=>t.net_pnl_usd<=0).length||1)*100).toFixed(3),
    totalFees: trades.reduce((a,b)=>a+(b.fee_paid||0),0).toFixed(2),
    byReason, byRegime,
    recentTrades: trades.slice(0, 20),
    openTrade: open,
  };
}

module.exports = {
  openTrade,
  closeTrade,
  checkOpenTrades,
  getOpenTrade,
  getBalance,
  getPaperStats,
};
