# Crypto Predictor Pro

Node/SQLite crypto prediction and paper-trading dashboard for BTC and ETH by default. It combines rule-based technical indicators, optional XGBoost inference, persistent learning feedback, risk gates, and simulated paper trades.

## Run

```powershell
npm install
npm start
```

Open:

- App: http://localhost:3000/crypto.html
- Status: http://localhost:3000/api/status
- Health: http://localhost:3000/api/health
- Active config: http://localhost:3000/api/config

## Configure

Create a private `config.json` from `config.example.json` and change only the values you want to override.

Common settings:

- `server.port`: backend port
- `coins`: supported coins and exchange symbols
- `trading.capitalUsd`: simulated capital used for risk sizing
- `paperTrading.startingBalanceUsd`: paper account starting balance per coin
- `risk`: stop loss, take profit, confidence, agreement, daily loss, and trade-count rules
- `marketData`: Binance base URL, candle interval, candle limit, and order book depth

`config.json` is ignored by git so local tuning stays private.

## Python ML

The app falls back to rule-based signals when no XGBoost model is available or Python cannot start. To point Node at a specific Python executable:

```powershell
$env:PYTHON_BIN="C:\Path\To\python.exe"
npm start
```

Train models after enough checked predictions exist:

```powershell
npm run train-btc
npm run train-eth
```

## Safety

This project is paper-trading only. Do not wire it to real exchange order placement without adding authenticated API handling, secrets management, dry-run safeguards, exchange-side risk limits, and much deeper backtesting.
