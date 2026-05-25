const fs = require('fs');
const path = require('path');

const DEFAULT_CONFIG = {
  server: {
    port: 3000,
  },
  marketData: {
    provider: 'binance',
    baseUrl: 'https://api.binance.com',
    candleInterval: '1m',
    candleLimit: 500,
    depthLimit: 20,
  },
  trading: {
    capitalUsd: 10000,
  },
  paperTrading: {
    startingBalanceUsd: 10000,
    timeoutMinutes: 15,
  },
  coins: {
    BTC: {
      symbol: 'BTCUSDT',
      slippageBasePct: 0.0002,
    },
    ETH: {
      symbol: 'ETHUSDT',
      slippageBasePct: 0.0004,
    },
  },
  risk: {
    stopLossPct: 0.005,
    takeProfitPct: 0.010,
    maxPositionPct: 0.10,
    feePct: 0.001,
    slippagePct: 0.0005,
    dailyLossLimitPct: 0.02,
    maxOpenPositions: 1,
    maxTradesPerDay: 20,
    minConfidence: 72,
    minAgreement: 0.80,
    minRegimes: ['bull', 'bear'],
    requireMACDCross: false,
    minExpectedEdgePct: 0.003,
    minDirectionalScore: 0.18,
    sidewaysOutcomePct: 0.0015,
    adaptiveConfidence: {
      enabled: true,
      lookback: 60,
      minSamples: 8,
      maxPenalty: 12,
      maxBonus: 4,
      targetWinRate: 0.55,
      minAvgPnlPct: 0.0005,
    },
  },
};

function isPlainObject(value) {
  return value && typeof value === 'object' && !Array.isArray(value);
}

function mergeDeep(base, override) {
  const output = { ...base };
  for (const [key, value] of Object.entries(override || {})) {
    if (isPlainObject(value) && isPlainObject(output[key])) {
      output[key] = mergeDeep(output[key], value);
    } else {
      output[key] = value;
    }
  }
  return output;
}

function loadLocalConfig() {
  const configPath = process.env.CONFIG_PATH || path.join(__dirname, 'config.json');
  if (!fs.existsSync(configPath)) return {};

  try {
    return JSON.parse(fs.readFileSync(configPath, 'utf8'));
  } catch (err) {
    console.warn(`[Config] Could not read ${configPath}: ${err.message}`);
    return {};
  }
}

const config = mergeDeep(DEFAULT_CONFIG, loadLocalConfig());
const coinSymbols = Object.fromEntries(
  Object.entries(config.coins).map(([coin, value]) => [coin, value.symbol])
);

function getCoinList() {
  return Object.keys(config.coins);
}

function getCoinSymbol(coin) {
  return config.coins[coin]?.symbol;
}

function getMarketUrl(pathname, params = {}) {
  const url = new URL(pathname, config.marketData.baseUrl);
  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, String(value));
  }
  return url.toString();
}

function validateConfig(activeConfig) {
  const errors = [];
  const requirePositiveNumber = (pathLabel, value) => {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
      errors.push(`${pathLabel} must be a positive number`);
    }
  };
  const requireNumberRange = (pathLabel, value, min, max) => {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
      errors.push(`${pathLabel} must be between ${min} and ${max}`);
    }
  };

  requirePositiveNumber('server.port', Number(activeConfig.server.port));
  requirePositiveNumber('marketData.candleLimit', activeConfig.marketData.candleLimit);
  requirePositiveNumber('marketData.depthLimit', activeConfig.marketData.depthLimit);
  requirePositiveNumber('trading.capitalUsd', activeConfig.trading.capitalUsd);
  requirePositiveNumber('paperTrading.startingBalanceUsd', activeConfig.paperTrading.startingBalanceUsd);
  requirePositiveNumber('paperTrading.timeoutMinutes', activeConfig.paperTrading.timeoutMinutes);

  try {
    new URL(activeConfig.marketData.baseUrl);
  } catch {
    errors.push('marketData.baseUrl must be a valid URL');
  }

  if (!isPlainObject(activeConfig.coins) || Object.keys(activeConfig.coins).length === 0) {
    errors.push('coins must contain at least one coin');
  } else {
    for (const [coin, coinConfig] of Object.entries(activeConfig.coins)) {
      if (!/^[A-Z0-9]{2,10}$/.test(coin)) {
        errors.push(`coins.${coin} must use an uppercase ticker key`);
      }
      if (!coinConfig.symbol || typeof coinConfig.symbol !== 'string') {
        errors.push(`coins.${coin}.symbol must be a string`);
      }
      requireNumberRange(`coins.${coin}.slippageBasePct`, coinConfig.slippageBasePct, 0, 0.05);
    }
  }

  requireNumberRange('risk.stopLossPct', activeConfig.risk.stopLossPct, 0, 0.5);
  requireNumberRange('risk.takeProfitPct', activeConfig.risk.takeProfitPct, 0, 0.5);
  requireNumberRange('risk.maxPositionPct', activeConfig.risk.maxPositionPct, 0, 1);
  requireNumberRange('risk.feePct', activeConfig.risk.feePct, 0, 0.1);
  requireNumberRange('risk.slippagePct', activeConfig.risk.slippagePct, 0, 0.1);
  requireNumberRange('risk.dailyLossLimitPct', activeConfig.risk.dailyLossLimitPct, 0, 1);
  requirePositiveNumber('risk.maxOpenPositions', activeConfig.risk.maxOpenPositions);
  requirePositiveNumber('risk.maxTradesPerDay', activeConfig.risk.maxTradesPerDay);
  requireNumberRange('risk.minConfidence', activeConfig.risk.minConfidence, 0, 100);
  requireNumberRange('risk.minAgreement', activeConfig.risk.minAgreement, 0, 1);
  requireNumberRange('risk.minExpectedEdgePct', activeConfig.risk.minExpectedEdgePct, 0, 1);
  requireNumberRange('risk.minDirectionalScore', activeConfig.risk.minDirectionalScore, 0, 1);
  requireNumberRange('risk.sidewaysOutcomePct', activeConfig.risk.sidewaysOutcomePct, 0, 1);
  if (activeConfig.risk.adaptiveConfidence) {
    requirePositiveNumber('risk.adaptiveConfidence.lookback', activeConfig.risk.adaptiveConfidence.lookback);
    requirePositiveNumber('risk.adaptiveConfidence.minSamples', activeConfig.risk.adaptiveConfidence.minSamples);
    requireNumberRange('risk.adaptiveConfidence.maxPenalty', activeConfig.risk.adaptiveConfidence.maxPenalty, 0, 50);
    requireNumberRange('risk.adaptiveConfidence.maxBonus', activeConfig.risk.adaptiveConfidence.maxBonus, 0, 50);
    requireNumberRange('risk.adaptiveConfidence.targetWinRate', activeConfig.risk.adaptiveConfidence.targetWinRate, 0, 1);
    requireNumberRange('risk.adaptiveConfidence.minAvgPnlPct', activeConfig.risk.adaptiveConfidence.minAvgPnlPct, -1, 1);
  }

  if (!Array.isArray(activeConfig.risk.minRegimes)) {
    errors.push('risk.minRegimes must be an array');
  }

  return errors;
}

const configErrors = validateConfig(config);
if (configErrors.length) {
  throw new Error(`Invalid config:\n- ${configErrors.join('\n- ')}`);
}

module.exports = {
  config,
  coinSymbols,
  getCoinList,
  getCoinSymbol,
  getMarketUrl,
  validateConfig,
};
