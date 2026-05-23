const express   = require('express');
const cors      = require('cors');
const fetch     = require('node-fetch');
const cron      = require('node-cron');
const path      = require('path');
const { db, stmts, insertCandlesBatch } = require('./database');
const risk      = require('./risk_manager');
const learning  = require('./learning_engine');
const paper     = require('./paper_trader');

const app  = express();
const PORT = 3000;
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// Serve crypto.html at root
app.get('/', (req, res) => res.sendFile(path.join(__dirname, 'crypto.html')));

const COINS = ['BTC', 'ETH'];
const SYMBOLS = { BTC: 'BTCUSDT', ETH: 'ETHUSDT' };

// ═══════════════════════════════════════════
// INDICATOR CALCULATIONS
// ═══════════════════════════════════════════
function calcEMA(p,n){const k=2/(Math.min(n,p.length)+1);let e=p.slice(0,Math.min(n,p.length)).reduce((a,b)=>a+b,0)/Math.min(n,p.length);for(let i=Math.min(n,p.length);i<p.length;i++)e=p[i]*k+e*(1-k);return e;}
function calcWilderRSI(p,n=14){if(p.length<n*2)return 50;let g=0,l=0;for(let i=1;i<=n;i++){const d=p[i]-p[i-1];d>0?g+=d:l+=Math.abs(d);}let ag=g/n,al=l/n;for(let i=n+1;i<p.length;i++){const d=p[i]-p[i-1];ag=(ag*(n-1)+(d>0?d:0))/n;al=(al*(n-1)+(d<0?Math.abs(d):0))/n;}return al===0?100:100-100/(1+ag/al);}
function calcStochRSI(p,rn=14,sn=14){if(p.length<rn+sn)return{k:50,d:50};const rv=[];for(let i=rn;i<=p.length;i++)rv.push(calcWilderRSI(p.slice(0,i),rn));if(rv.length<sn)return{k:50,d:50};const sl=rv.slice(-sn),mn=Math.min(...sl),mx=Math.max(...sl);const k=mx===mn?50:((rv[rv.length-1]-mn)/(mx-mn))*100;return{k,d:rv.slice(-3).reduce((a,b)=>a+b,0)/3};}
function calcWR(p,n=14){if(p.length<n)return-50;const sl=p.slice(-n),h=Math.max(...sl),l=Math.min(...sl);return h===l?-50:((h-p[p.length-1])/(h-l))*-100;}
function calcCCI(c,n=20){if(c.length<n)return 0;const sl=c.slice(-n),tps=sl.map(x=>(x.high+x.low+x.close)/3),mean=tps.reduce((a,b)=>a+b,0)/n,md=tps.reduce((a,b)=>a+Math.abs(b-mean),0)/n;return md===0?0:(tps[tps.length-1]-mean)/(0.015*md);}
function calcATR(c,n=14){if(c.length<n+1)return null;const trs=c.slice(1).map((x,i)=>Math.max(x.high-x.low,Math.abs(x.high-c[i].close),Math.abs(x.low-c[i].close)));return trs.slice(-n).reduce((a,b)=>a+b,0)/n;}
function calcVWAP(c){if(!c.length)return null;let tv=0,tvp=0;c.forEach(x=>{const tp=(x.high+x.low+x.close)/3;tvp+=tp*x.volume;tv+=x.volume;});return tv>0?tvp/tv:null;}
function calcADX(c,n=14){if(c.length<n*2)return{adx:0,diPlus:0,diMinus:0};const tr=[],dp=[],dm=[];for(let i=1;i<c.length;i++){const x=c[i],p=c[i-1];tr.push(Math.max(x.high-x.low,Math.abs(x.high-p.close),Math.abs(x.low-p.close)));const u=x.high-p.high,d=p.low-x.low;dp.push(u>d&&u>0?u:0);dm.push(d>u&&d>0?d:0);}const atr=tr.slice(-n).reduce((a,b)=>a+b,0)/n,adp=dp.slice(-n).reduce((a,b)=>a+b,0)/n,adm=dm.slice(-n).reduce((a,b)=>a+b,0)/n;const dip=atr>0?(adp/atr)*100:0,dim=atr>0?(adm/atr)*100:0;return{adx:dip+dim>0?Math.abs(dip-dim)/(dip+dim)*100:0,diPlus:dip,diMinus:dim};}
function calcBB(p,n=20,m=2){if(p.length<n)return null;const sl=p.slice(-n),mean=sl.reduce((a,b)=>a+b,0)/n,std=Math.sqrt(sl.reduce((a,b)=>a+(b-mean)**2,0)/n);return{upper:mean+m*std,middle:mean,lower:mean-m*std,std,bw:(std*2*m)/mean*100};}
function calcMACD(p,f=12,s=26,sg=9){if(p.length<s+sg)return null;const kf=2/(f+1),ks=2/(s+1);let ef=p.slice(0,f).reduce((a,b)=>a+b,0)/f,es=p.slice(0,s).reduce((a,b)=>a+b,0)/s;const ml=[];for(let i=s;i<p.length;i++){ef=p[i]*kf+ef*(1-kf);es=p[i]*ks+es*(1-ks);ml.push(ef-es);}if(ml.length<sg)return null;const ksg=2/(sg+1);let sv=ml.slice(0,sg).reduce((a,b)=>a+b,0)/sg;const out=[];for(let i=sg;i<ml.length;i++){sv=ml[i]*ksg+sv*(1-ksg);out.push({macd:ml[i],signal:sv,hist:ml[i]-sv});}return out;}
function calcCVD(c){if(c.length<5)return 0;const r=c.slice(-30),d=r.map(x=>{const rt=(x.close-x.low)/(x.high-x.low||1);return x.volume*(rt*2-1);});const cvd=d.reduce((a,b)=>a+b,0),avg=r.reduce((a,b)=>a+b.volume,0)/r.length;return Math.max(-1,Math.min(1,cvd/(avg*r.length)));}
function detectRegime(prices,candles){
  if(prices.length<50)return'neutral';
  const adx=calcADX(candles),e20=calcEMA(prices,20),e50=calcEMA(prices,Math.min(50,prices.length)),last=prices[prices.length-1];
  const rets=prices.slice(-20).map((p,i)=>i>0?(p-prices[prices.length-21+i])/prices[prices.length-21+i]:0).slice(1);
  const vol=Math.sqrt(rets.reduce((a,b)=>a+b*b,0)/rets.length)*100;
  if(vol>1.5)return'volatile';
  if(e20>e50&&last>e20&&adx.adx>25)return'bull';
  if(e20<e50&&last<e20&&adx.adx>25)return'bear';
  if(adx.adx<20)return'range';
  return'neutral';
}
function clamp(v,a,b){return Math.max(a,Math.min(b,v));}

// ── Build feature vector ─────────────────────────────────
function buildFeatures(coin, prices, candles, ob) {
  const rsi=calcWilderRSI(prices),stoch=calcStochRSI(prices),wr=calcWR(prices),cci=calcCCI(candles);
  const e9=calcEMA(prices,9),e21=calcEMA(prices,21),e50=calcEMA(prices,Math.min(50,prices.length));
  const macd=calcMACD(prices),lastM=macd&&macd.length>0?macd[macd.length-1]:{macd:0,signal:0,hist:0};
  const bb=calcBB(prices)||{upper:0,middle:0,lower:0,bw:0};
  const atr=calcATR(candles)||0,adxObj=calcADX(candles),vwap=calcVWAP(candles)||0,cvd=calcCVD(candles);
  const last=prices[prices.length-1];
  const mom5=prices.length>5?((last-prices[prices.length-6])/prices[prices.length-6])*100:0;
  const mom10=prices.length>10?((last-prices[prices.length-11])/prices[prices.length-11])*100:0;
  // Volume delta
  let bv=0,sv=0;candles.slice(-20).forEach(c=>{const r=(c.close-c.low)/(c.high-c.low||1);bv+=c.volume*r;sv+=c.volume*(1-r);});
  const vdBuyPct=(bv/(bv+sv||1))*100;
  // OB imbalance
  let obL5=0,obL10=0,obL20=0;
  if(ob&&ob.bids&&ob.asks){
    const imb=(n)=>{const bvol=ob.bids.slice(0,n).reduce((a,b)=>a+parseFloat(b[0])*parseFloat(b[1]),0),avol=ob.asks.slice(0,n).reduce((a,b)=>a+parseFloat(b[0])*parseFloat(b[1]),0),t=bvol+avol||1;return((bvol-avol)/t)*100;};
    obL5=imb(5);obL10=imb(10);obL20=imb(20);
  }
  // Spike
  const spikeRatio=candles.length>=20?candles[candles.length-1].volume/(candles.slice(-20,-1).reduce((a,b)=>a+b.volume,0)/19):1;
  const tod=getTimeOfDay();
  const regime=detectRegime(prices,candles);
  // Patterns
  let bullW=0,bearW=0;
  if(candles.length>=3){
    const c=candles[candles.length-1],p=candles[candles.length-2];
    const body=Math.abs(c.close-c.open),range=c.high-c.low||0.001,bull=c.close>c.open,pb=p.close>p.open,pvb=Math.abs(p.close-p.open);
    if(!pb&&bull&&c.open<p.close&&c.close>p.open&&body>pvb)bullW+=3;
    if(pb&&!bull&&c.open>p.close&&c.close<p.open&&body>pvb)bearW+=3;
    if(body/range>0.9&&bull)bullW+=3;if(body/range>0.9&&!bull)bearW+=3;
  }
  return {
    symbol:coin,timestamp:Date.now(),
    rsi,wilder_rsi:rsi,stoch_k:stoch.k,stoch_d:stoch.d,
    williams_r:wr,cci,ema9:e9,ema21:e21,ema50:e50,
    macd:lastM.macd,macd_signal:lastM.signal,macd_hist:lastM.hist,
    bb_upper:bb.upper,bb_middle:bb.middle,bb_lower:bb.lower,bb_bw:bb.bw,
    atr,adx:adxObj.adx,di_plus:adxObj.diPlus,di_minus:adxObj.diMinus,
    vwap,cvd,ob_imb_l5:obL5,ob_imb_l10:obL10,ob_imb_l20:obL20,
    vol_buy_pct:vdBuyPct,vol_sell_pct:100-vdBuyPct,
    regime,tod_session:tod.session,tod_score:tod.score,
    mom5,mom10,pattern_bull_w:bullW,pattern_bear_w:bearW,vol_spike_ratio:spikeRatio,
  };
}

function getTimeOfDay(){
  const t=new Date().getUTCHours()+new Date().getUTCMinutes()/60;
  if(t>=6&&t<8)return{session:'London Open',score:0.3};
  if(t>=12&&t<14)return{session:'NY Open',score:0.4};
  if(t>=14&&t<17)return{session:'NY Peak',score:0.2};
  if(t>=17&&t<20)return{session:'NY Close',score:-0.2};
  return{session:'Off-Peak',score:-0.1};
}

// ── Generate rule-based signal (used before XGB ready) ───
function generateRuleSignal(prices, candles, weights) {
  const rsi=calcWilderRSI(prices),e9=calcEMA(prices,9),e21=calcEMA(prices,21);
  const stoch=calcStochRSI(prices),macd=calcMACD(prices);
  const bb=calcBB(prices),vwap=calcVWAP(candles);
  const adx=calcADX(candles),last=prices[prices.length-1];
  const mom5=prices.length>5?((last-prices[prices.length-6])/prices[prices.length-6])*100:0;

  let score=0,totalW=0,agreements=0,total=0;
  const indSignals={};

  const add=(name,sig,w)=>{score+=sig*w;totalW+=w;total++;indSignals[name]=sig>=0?'UP':'DOWN';if(Math.sign(sig)===Math.sign(score)||total===1)agreements++;};

  add('wilderRsi', rsi<35?1:rsi>65?-1:(50-rsi)/20, weights.wilderRsi||2.5);
  add('emaCross',  e9>e21?1:-1, weights.emaCross||2.5);
  if(macd&&macd.length>1){const lm=macd[macd.length-1],pm=macd[macd.length-2];add('macd',lm.hist>0?0.8:-0.8,weights.macd||1.5);if(pm.macd<pm.signal&&lm.macd>=pm.signal){add('macdCross',1,weights.macdCross||2);}else if(pm.macd>pm.signal&&lm.macd<=pm.signal){add('macdCross',-1,weights.macdCross||2);}}
  if(bb){add('bb',last<bb.lower?1:last>bb.upper?-1:0,weights.bb||1.5);}
  add('stochRsi',stoch.k<20?1:stoch.k>80?-1:0,weights.stochRsi||2);
  if(adx.adx>20){add('adx',adx.diPlus>adx.diMinus?0.6:-0.6,weights.adx||1.5);}
  if(vwap){add('vwap',last>vwap?0.5:-0.5,weights.vwap||1.2);}
  add('mom',clamp(mom5/0.4,-1,1),weights.mom||1);

  const rawScore=totalW>0?score/totalW:0;
  const signal=rawScore>=0?'UP':'DOWN';
  const agreement=total>0?agreements/total:0;
  const confidence=Math.round(clamp(50+Math.abs(rawScore)*55,20,97));
  return{signal,confidence,rawScore,agreement,indSignals};
}

// ── Call Python XGBoost ──────────────────────────────────
function runXGBoost(coin, featuresObj) {
  return new Promise((resolve) => {
    const { spawn } = require('child_process');
    const proc = spawn('python', ['ml_predict.py'], { cwd: __dirname });
    let out='';
    proc.stdout.on('data',d=>out+=d.toString());
    proc.stderr.on('data',()=>{});
    proc.on('close',()=>{try{resolve(JSON.parse(out));}catch{resolve({error:'parse',signal:null,confidence:0});}});
    proc.stdin.write(JSON.stringify({symbol:coin,features:featuresObj}));
    proc.stdin.end();
  });
}

// ── In-memory market state ────────────────────────────────
const marketState = {
  BTC:{prices:[],candles:[],ob:null},
  ETH:{prices:[],candles:[],ob:null},
};

// ── Pending prediction tracker (5-min window) ───────────
const pendingPreds = { BTC: null, ETH: null };

async function checkPendingPredictions() {
  const now = Date.now();
  for (const coin of COINS) {
    const p = pendingPreds[coin];
    if (!p || p.checked) continue;
    if (now < p.checkAt) continue; // not 5 min yet

    try {
      const res  = await fetch(`https://api.binance.com/api/v3/ticker/price?symbol=${SYMBOLS[coin]}`);
      const data = await res.json();
      const exitPrice = parseFloat(data.price);
      const pnlPct  = p.signal === 'UP'
        ? (exitPrice - p.entryPrice) / p.entryPrice
        : (p.entryPrice - exitPrice) / p.entryPrice;
      const netPnl  = pnlPct - (risk.DEFAULT_CONFIG.feePct * 2); // subtract round-trip fee
      const correct = netPnl > 0;

      learning.recordOutcome(coin, p.predId, p.signal, p.indSignals, p.entryPrice, exitPrice, correct, netPnl, p.regime);
      risk.onTradeClose(coin, netPnl);
      p.checked = true;

      console.log(`[5m Check] ${coin} ${p.signal} entry:${p.entryPrice.toFixed(2)} exit:${exitPrice.toFixed(2)} net:${(netPnl*100).toFixed(3)}% ${correct?'✓':'✗'}`);
    } catch(e) {
      console.error(`[5m Check] ${coin} error:`, e.message);
    }
  }
}

// ═══════════════════════════════════════════
// API ROUTES
// ═══════════════════════════════════════════

// GET /api/predict/:coin
app.get('/api/predict/:coin', async (req, res) => {
  const coin = req.params.coin.toUpperCase();
  const s    = marketState[coin];
  if (!s || s.prices.length < 50)
    return res.json({ error:'not_ready', message:'Loading market data...' });

  const price    = s.prices[s.prices.length - 1];
  const regime   = detectRegime(s.prices, s.candles);
  const weights  = learning.loadWeights(coin);
  const ruleSig  = generateRuleSignal(s.prices, s.candles, weights);
  const features = buildFeatures(coin, s.prices, s.candles, s.ob);

  // Persist features
  let featureId = null;
  try {
    const featRow = stmts.insertFeatures.run(features);
    featureId = featRow.lastInsertRowid;
  } catch(e) {}

  // XGBoost (if model exists)
  const xgb = await runXGBoost(coin, features);

  // Final signal: prefer XGBoost if available and confident
  let finalSignal = ruleSig.signal;
  let finalConf   = ruleSig.confidence;
  let modelUsed   = 'rule';

  if (xgb && !xgb.error && xgb.signal && xgb.confidence > 55) {
    // Agree with rule-based? Boost confidence
    if (xgb.signal === ruleSig.signal) {
      finalSignal = xgb.signal;
      finalConf   = Math.min(97, Math.round((xgb.confidence + ruleSig.confidence) / 2 + 5));
      modelUsed   = 'ensemble';
    } else {
      // Disagree — use whichever is more confident, but reduce confidence
      finalSignal = xgb.confidence > ruleSig.confidence ? xgb.signal : ruleSig.signal;
      finalConf   = Math.round(Math.max(xgb.confidence, ruleSig.confidence) * 0.85);
      modelUsed   = 'conflicted';
    }
  }

  // Risk check
  const riskCheck = risk.shouldTrade(coin, finalSignal, finalConf, ruleSig.agreement, regime, price, 10000);

  // Kelly sizing
  const pnlStats  = stmts.getStats.get({ symbol: coin });
  const hist      = stmts.getRecentPredictions.all({ symbol: coin, limit: 50 });
  const wins      = hist.filter(h=>h.correct===1);
  const losses    = hist.filter(h=>h.correct===0);
  const winRate   = hist.length>0?wins.length/hist.length:0.5;
  const avgWin    = wins.length>0?wins.reduce((a,b)=>a+Math.abs(b.result_pnl||0),0)/wins.length:0.005;
  const avgLoss   = losses.length>0?losses.reduce((a,b)=>a+Math.abs(b.result_pnl||0),0)/losses.length:0.005;
  const kellySz   = risk.kellySize(10000, winRate, avgWin, avgLoss);

  // Save prediction to DB
  let predId = null;
  if (featureId) {
    try {
      const predRow = stmts.insertPrediction.run({
        symbol:coin, timestamp:Date.now(), price,
        signal:finalSignal, confidence:finalConf,
        prob_up:   xgb?.probs?.UP   || (finalSignal==='UP'?finalConf:100-finalConf),
        prob_down: xgb?.probs?.DOWN || (finalSignal==='DOWN'?finalConf:100-finalConf),
        prob_side: 0,
        ensemble_mom: ruleSig.rawScore,
        ensemble_mr: 0, ensemble_vol: 0,
        model_version: modelUsed,
        feature_id: featureId,
      });
      predId = predRow.lastInsertRowid;
    } catch(e) {}
  }

  // Open paper trade if risk allows
  if (riskCheck.allowed && predId) {
    const openTrade = paper.getOpenTrade(coin);
    if (!openTrade) {
      paper.openTrade(coin, finalSignal, price, finalConf, ruleSig.agreement, regime, modelUsed, predId);
      risk.onTradeOpen(coin);
    }
  }

  // Register pending 5-min check ONLY if risk allows
  if (riskCheck.allowed && predId && !pendingPreds[coin]?.checked === false) {
    pendingPreds[coin] = {
      predId, signal: finalSignal, entryPrice: price,
      checkAt: Date.now() + 5 * 60 * 1000,
      indSignals: ruleSig.indSignals, regime, checked: false,
    };
    risk.onTradeOpen(coin);
    console.log(`[Trade] ${coin} ${finalSignal} @ $${price.toFixed(2)} conf:${finalConf}% — check at ${new Date(Date.now()+5*60*1000).toLocaleTimeString()}`);
  }

  const session = risk.getSessionSummary(coin);

  res.json({
    coin, price, signal: finalSignal, confidence: finalConf,
    model: modelUsed,
    riskAllowed: riskCheck.allowed,
    riskReasons: riskCheck.reasons,
    kellySizeUSD: kellySz.toFixed(2),
    probs: xgb?.probs || { UP: finalSignal==='UP'?finalConf:100-finalConf, DOWN: finalSignal==='DOWN'?finalConf:100-finalConf },
    agreement: ruleSig.agreement,
    xgb: xgb?.error ? null : xgb,
    features: { rsi:features.wilder_rsi, ema_trend:features.ema9>features.ema21?'bull':'bear', regime, cvd:features.cvd, tod:features.tod_session },
    session,
    pred_id: predId,
  });
});

// GET /api/paper/:coin
app.get('/api/paper/:coin', (req, res) => {
  const coin = req.params.coin.toUpperCase();
  res.json(paper.getPaperStats(coin));
});

// GET /api/learning/:coin
app.get('/api/learning/:coin', (req, res) => {
  const coin = req.params.coin.toUpperCase();
  res.json(learning.getLearningStats(coin));
});

// GET /api/history/:coin
app.get('/api/history/:coin', (req, res) => {
  const coin  = req.params.coin.toUpperCase();
  const limit = parseInt(req.query.limit) || 50;
  const preds = stmts.getRecentPredictions.all({ symbol: coin, limit });
  const stats = stmts.getStats.get({ symbol: coin });
  const model = stmts.getModelLog.all({ symbol: coin });
  res.json({ predictions: preds, stats, modelLog: model });
});

// GET /api/risk/:coin
app.get('/api/risk/:coin', (req, res) => {
  const coin = req.params.coin.toUpperCase();
  res.json(risk.getSessionSummary(coin));
});

// GET /api/status
app.get('/api/status', (req, res) => {
  const fs = require('fs');
  const out = {};
  for (const coin of COINS) {
    const s     = marketState[coin];
    const count = stmts.countTrainingData.get({ symbol: coin });
    out[coin] = {
      prices_loaded:    s.prices.length,
      training_samples: count.count,
      model_exists:     fs.existsSync(path.join(__dirname, `model_${coin}.json`)),
      last_price:       s.prices[s.prices.length-1] || null,
      pending_pred:     pendingPreds[coin] ? {
        signal: pendingPreds[coin].signal,
        entry:  pendingPreds[coin].entryPrice,
        checksIn: Math.max(0, Math.round((pendingPreds[coin].checkAt - Date.now())/1000)) + 's',
      } : null,
      session: risk.getSessionSummary(coin),
    };
  }
  res.json(out);
});

// POST /api/train/:coin
app.post('/api/train/:coin', (req, res) => {
  const coin = req.params.coin.toUpperCase();
  res.json({ status:'started', coin });
  learning.trainXGBoost(coin);
});

// POST /api/walkforward/:coin
app.post('/api/walkforward/:coin', (req, res) => {
  const coin = req.params.coin.toUpperCase();
  const wf   = learning.runWalkForward(coin);
  res.json(wf || { error: 'not_enough_data' });
});

// ═══════════════════════════════════════════
// MARKET DATA FETCH
// ═══════════════════════════════════════════
async function fetchMarketData(coin) {
  const sym = SYMBOLS[coin];
  try {
    const [klRes, obRes] = await Promise.all([
      fetch(`https://api.binance.com/api/v3/klines?symbol=${sym}&interval=1m&limit=500`),
      fetch(`https://api.binance.com/api/v3/depth?symbol=${sym}&limit=20`),
    ]);
    const klines = await klRes.json();
    const ob     = await obRes.json();
    marketState[coin].prices  = klines.map(k => parseFloat(k[4]));
    marketState[coin].candles = klines.map(k => ({ open:parseFloat(k[1]),high:parseFloat(k[2]),low:parseFloat(k[3]),close:parseFloat(k[4]),volume:parseFloat(k[5]) }));
    marketState[coin].ob = ob;
    // Persist candles
    insertCandlesBatch(klines.map(k => ({ symbol:coin, interval:'1m', open_time:parseInt(k[0]), open:parseFloat(k[1]),high:parseFloat(k[2]),low:parseFloat(k[3]),close:parseFloat(k[4]),volume:parseFloat(k[5]) })));
  } catch(e) { console.error(`[Fetch] ${coin}:`, e.message); }
}

// ═══════════════════════════════════════════
// CRON JOBS
// ═══════════════════════════════════════════

// Market data every 30s + check open paper trades
cron.schedule('*/30 * * * * *', async () => {
  for (const coin of COINS) {
    await fetchMarketData(coin);
    const s = marketState[coin];
    if (s.candles.length > 0) {
      const last = s.candles[s.candles.length - 1];
      await paper.checkOpenTrades(coin, last.close, last.high, last.low);
    }
  }
});

// Check pending 5-min predictions every 30s
cron.schedule('*/30 * * * * *', checkPendingPredictions);

// Walk-forward validation daily at midnight
cron.schedule('0 0 * * *', () => {
  for (const coin of COINS) {
    console.log(`[WF] Running walk-forward for ${coin}`);
    learning.runWalkForward(coin);
  }
});

// ═══════════════════════════════════════════
// START
// ═══════════════════════════════════════════
async function start() {
  console.log('╔══════════════════════════════════════╗');
  console.log('║   Crypto Predictor PRO — Backend v2  ║');
  console.log('╚══════════════════════════════════════╝');
  for (const coin of COINS) {
    console.log(`Loading ${coin}...`);
    await fetchMarketData(coin);
    // Bootstrap learning weights if not in DB
    const w = learning.loadWeights(coin);
    console.log(`${coin} weights loaded (${Object.keys(w).length} features)`);
  }
  app.listen(PORT, () => {
    console.log(`\n✅ Server at http://localhost:${PORT}`);
    console.log(`   App:    http://localhost:${PORT}/crypto.html`);
    console.log(`   Status: http://localhost:${PORT}/api/status\n`);
  });
}
start();
