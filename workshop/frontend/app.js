/* =================== Config =================== */
const API = "/api";
/* Dica: se você for servir o HTML em outra origem (ex.: porta diferente do backend),
   troque para, por exemplo:
   const API = "http://localhost:8080/api";
*/

/* -------- robust network helper (JSON-or-text) -------- */
async function safeFetch(url, opts){
  const res = await fetch(url, Object.assign({ cache: 'no-cache' }, (opts||{})));
  const text = await res.text();
  let json = null;
  try { json = JSON.parse(text); } catch(e) {}
  if (json !== null) {
    if (!res.ok) {
      console.error("API error:", url, json);
      throw new Error(json.error || json.detail || (json.meta && json.meta.error) || res.statusText || "API error");
    }
    return json;
  }
  console.error("Non-JSON response from", url, res.status, text.slice(0,300));
  throw new Error("API returned non-JSON (status " + res.status + ")");
}

const POLL_MS_CANDLES = 1500;   // 1.5s
const POLL_MS_SIGNALS = 5000;   // 5s
const MAX_MARKERS = 400;

const FEE_PERC = 0.0002;
const SLIPPAGE_PERC = 0.0;
const TF_SEC = { '1m':60, '5m':300, '15m':900 };

const FUTURE_OFFSET_BARS = 50;

/* ====== Fonte dos sinais ====== */
const SOURCE_MODE_DEFAULT = 'live';
const HOURS_LOOKBACK_DEFAULT = 24*30; // 720 h

/* ====== Execucao ====== */
const ALLOW_OVERLAP = false;
const HOLD_SCALE = 1.0;

/* ====== TP progressivo (referência visual + simulação) ====== */
const USE_PROG_TP_DEFAULT = true;
const TP_SCALE = 1.0;
const PROG_TP_DEFAULTS = {
  enabled: true,
  tp_perc:   [0.003, 0.008, 0.015],
  tp_shares: [0.40,  0.40,  0.20],
  sl_mode: 'fixed',        // 'fixed'|'entry'|'atr' (front usa fixed como fallback)
  sl_fixed_perc: 0.003,    // usado apenas se explicitamente ligado
  atr_len: 14,
  atr_mult: 1.0,
  step_on_tp: ['entry','tp1','tp1']
};

/* ====== Execução financeira (notional) ====== */
const DEFAULT_CONTRACTS = 1.0;
const DEFAULT_CONTRACT_VALUE = 1.0; // 1 BTC por contrato (USDT-margined)

/* =================== Flags de debug =================== */
const DEBUG_SANITIZE = false;

/* =================== Fallbacks de robustez =================== */
const MIN_INIT_BARS = 80;        // mínimo de candles para considerar "carregado"
const STALE_RELOAD_MULT = 3;     // se ficar > 3 barras sem novo candle, recarrega full

/* =================== State =================== */
const state = {
  symbol: 'BTCUSDT',
  tf: '5m',
  viewType: 'base',
  methodId: null,
  methodMap: new Map(),
  currency: 'USD',
  fx: 1,

  sourceMode: SOURCE_MODE_DEFAULT,
  hoursLookback: HOURS_LOOKBACK_DEFAULT,

  priceChart: null,
  volumeChart: null,
  candleSeries: null,
  volumeSeries: null,

  lastTs: 0,
  baseCandles: null,
  signals: [],
  tradeLines: [],
  tradeIndexMap: new Map(),
  rrDefault: 2.0,
  timeouts: { '1m':200, '5m':100, '15m':50 },

  followRealtime: true,
  _markersCache: [],

  // Execução notional
  contracts: DEFAULT_CONTRACTS,
  contractValue: DEFAULT_CONTRACT_VALUE,

  // TP progressivo
  useProgTP: USE_PROG_TP_DEFAULT,

  // Execução de custos (podem vir do backend)
  feePerc: FEE_PERC,
  slippagePerc: SLIPPAGE_PERC,
};

/* =================== Utils =================== */
function fmt(n, p){ return Number(n).toFixed(p == null ? 2 : p); }
function toSec(ms){ return Math.floor(ms/1000); }
function nowSec(){ return Math.floor(Date.now()/1000); }
function stepSec(){ return TF_SEC[state.tf] || 60; }
function isFiniteNumber(x){ return x !== null && x !== '' && Number.isFinite(Number(x)); }
function notionalMult(){ return (Number(state.contracts)||0) * (Number(state.contractValue)||0); }

/* === Helpers para ler o config do método e timeout === */
function getActiveMethodConfig(){
  const m = (state.methodMap && state.methodId) ? state.methodMap.get(state.methodId) : null;
  return (m && m.config) ? m.config : {};
}
function buildTimeoutCfgFromMethod(){
  const cfg = getActiveMethodConfig();
  const m = { ...state.timeouts };
  if (cfg && cfg.timeout_mode === 'bars' && isFiniteNumber(cfg.max_hold)) {
    m[state.tf] = Number(cfg.max_hold);
  }
  return { max_hold_map: m };
}

/* Persistência de execução (contracts/contractValue/useProgTP) */
const execStoreKey = () => "SCALP_EXEC";
function loadExecSettings(){
  try{
    const s = localStorage.getItem(execStoreKey());
    if (!s) return;
    const j = JSON.parse(s);
    if (isFiniteNumber(j.contracts)) state.contracts = Number(j.contracts);
    if (isFiniteNumber(j.contractValue)) state.contractValue = Number(j.contractValue);
    if (typeof j.useProgTP === 'boolean') state.useProgTP = j.useProgTP;
  }catch(e){}
}
function saveExecSettings(){
  try{
    localStorage.setItem(execStoreKey(), JSON.stringify({
      contracts: state.contracts,
      contractValue: state.contractValue,
      useProgTP: state.useProgTP
    }));
  }catch(e){}
}

/* 30 dias de barras por TF */
function candlesLimitForTF(tf){
  if (tf === '1m') return 45000;
  if (tf === '5m') return 9000;
  if (tf === '15m') return 3000;
  return 1000;
}

const storeKey = () => "SCALPTV_" + state.symbol + "_" + state.tf + "_" + state.viewType + "_" + (state.methodId||"__");
const CACHE_TTL_MS = 5*60*1000;
function saveCache(payload){
  try { localStorage.setItem(storeKey(), JSON.stringify({ ...payload, savedAt: Date.now() })); } catch (e) {}
}
function loadCache(){
  try{
    const s = localStorage.getItem(storeKey());
    if (!s) return null;
    const j = JSON.parse(s);
    if ((Date.now() - (j.savedAt||0)) > CACHE_TTL_MS) return null;
    return j;
  } catch (e) { return null; }
}

/* =================== ATR util (para quando o JSON pedir ATR stop/TP) =================== */
function buildATRMap(candles, len){
  len = Math.max(1, Number(len) || 14);
  const out = new Map();
  if (!Array.isArray(candles) || candles.length === 0) return out;

  let prevClose = candles[0].close;
  const TR = [];
  let atrPrev = null;

  for (let i = 0; i < candles.length; i++){
    const c = candles[i];
    const tr = Math.max(
      c.high - c.low,
      Math.abs(c.high - prevClose),
      Math.abs(c.low - prevClose)
    );
    TR.push(tr);
    prevClose = c.close;

    if (i + 1 === len){
      const sma = TR.reduce((a,b)=>a+b,0) / len;
      atrPrev = sma;
      out.set(c.time, sma);
    } else if (i + 1 > len){
      atrPrev = (atrPrev * (len - 1) + tr) / len; // Wilder
      out.set(c.time, atrPrev);
    } else {
      out.set(c.time, null);
    }
  }
  return out;
}
function atrAt(atrMap, tSec){
  if (!atrMap || atrMap.size === 0) return null;
  let val = null;
  for (const [t, v] of atrMap) {
    if (t > tSec) break;
    if (v != null) val = v;
  }
  return val;
}

/* =================== Fallbacks de stops/TP progressivo =================== */
/* Agora: só aplica stop fixo se houver percentual explícito no cfg (sem fallback 0,3%) */
function initialStop(entry, dir, cfg){
  const perc = (cfg && isFiniteNumber(cfg.sl_fixed_perc)) ? Number(cfg.sl_fixed_perc) : null;
  if (perc == null) return null;
  return dir > 0 ? (entry * (1 - perc)) : (entry * (1 + perc));
}
function buildProgressiveTargets(entry, dir, cfg){
  const arr = (cfg && Array.isArray(cfg.tp_perc) ? cfg.tp_perc : [0.003,0.008,0.015]);
  return arr.map(p => dir>0 ? entry * (1 + p*TP_SCALE) : entry * (1 - p*TP_SCALE));
}
function moveStopOnTP(prevStop, entry, tps, hitIdx, stepPolicy, dir){
  const sp = Array.isArray(stepPolicy) && stepPolicy.length ? stepPolicy : ['entry','tp1','tp1'];
  const label = sp[Math.min(hitIdx, sp.length-1)];
  if (label === 'entry') return entry;
  const m = /^tp(\d+)$/i.exec(label);
  if (m){
    const k = Math.max(1, Math.min(tps.length, Number(m[1])));
    return tps[k-1];
  }
  return prevStop;
}

/* =================== Sanitização de candles =================== */
function mapCandleRaw(c){
  let t = c.time;
  if (t instanceof Date) t = Math.floor(t.getTime() / 1000);
  else if (typeof t === 'string') {
    const parsed = Date.parse(t);
    t = isNaN(parsed) ? Math.floor(Date.now()/1000) : Math.floor(parsed/1000);
  } else {
    t = Number(t);
    if (isNaN(t)) t = Math.floor(Date.now()/1000);
    if (t > 1e12) t = Math.floor(t / 1000); // ms->sec
  }
  return {
    time: t,
    open: +c.open, high: +c.high, low: +c.low, close: +c.close,
    volume: +c.volume
  };
}

function isValidCandle(c){
  if (!c) return false;
  const t = Number(c.time);
  const o = Number(c.open);
  const h = Number(c.high);
  const l = Number(c.low);
  const cl= Number(c.close);
  const v = Number(c.volume);
  if (!Number.isFinite(t) || t <= 0) return false;
  if (![o,h,l,cl,v].every(Number.isFinite)) return false;
  const hi = Math.max(h, l);
  const lo = Math.min(h, l);
  if (hi < lo) return false;
  if (o < lo || o > hi) return false;
  if (cl< lo || cl> hi) return false;
  if (v < 0) return false;
  return true;
}

function normalizeForSeries(c, fx){
  if (!c) return null;
  let t = Number(c.time);
  let o = Number(c.open);
  let h = Number(c.high);
  let l = Number(c.low);
  let cl= Number(c.close);
  let v = Number(c.volume);

  if (![t,o,h,l,cl,v].every(Number.isFinite) || t <= 0) return null;
  if (h < l) { const tmp = h; h = l; l = tmp; }
  if (o  < l) o  = l; if (o  > h) o  = h;
  if (cl < l) cl = l; if (cl > h) cl = h;
  if (!Number.isFinite(v) || v < 0) v = 0;

  return {
    time: t,
    open: o * fx, high: h * fx, low: l * fx, close: cl * fx,
    baseOpen: o, baseClose: cl,
    volume: v
  };
}

/* Ordena e remove duplicatas por tempo */
function sanitizeCandles(rawArr){
  const mapped = (Array.isArray(rawArr) ? rawArr : []).map(mapCandleRaw);
  mapped.sort((a,b) => a.time - b.time);
  const uniq = [];
  let lastT = -1;
  for (let i=0;i<mapped.length;i++){
    const c = mapped[i];
    if (!Number.isFinite(c.time)) continue;
    if (c.time === lastT) {
      uniq[uniq.length - 1] = c;
    } else {
      uniq.push(c);
      lastT = c.time;
    }
  }
  return uniq;
}

/* =================== API =================== */
async function fetchCandles(limit, since) {
  const url = new URL(API + "/candles", window.location.origin);
  url.searchParams.set("symbol", state.symbol);
  url.searchParams.set("tf", state.tf);
  url.searchParams.set("limit", limit == null ? candlesLimitForTF(state.tf) : limit);
  if (since != null) url.searchParams.set("since", since*1000); // ms
  try {
    const j = await safeFetch(url.toString());
    return Array.isArray(j.candles) ? j.candles : [];
  } catch (e) {
    console.error("candles fetch failed:", e);
    return []; // fallback: mantém UI viva mesmo com 502
  }
}
async function fetchFX() {
  try {
    const j = await safeFetch(API + "/fx?pair=USDTBRL");
    const px = (j && typeof j.price !== 'undefined' && j.price !== null) ? j.price : 1;
    state.fx = Number(px) || 1;
  } catch (e) {
    console.warn("fx fetch failed:", e);
    state.fx = 1; // fallback seguro
  }
  const el = document.getElementById('fxpill');
  if (el) el.textContent = "USDTBRL  " + fmt(state.fx,2);
}
async function fetchMethods() {
  const url = API + "/methods?type=" + state.viewType + "&tf=" + state.tf + "&with=metrics,config,params";
  return await safeFetch(url);
}
async function fetchSignalsFor(methodId) {
  const url = new URL(API + "/signals", window.location.origin);
  url.searchParams.set("symbol", state.symbol);
  url.searchParams.set("tf", state.tf);
  url.searchParams.set("type", state.viewType);
  url.searchParams.set("hours", String(state.hoursLookback || HOURS_LOOKBACK_DEFAULT));
  if (methodId) url.searchParams.set("id", methodId);
  url.searchParams.set("_", Date.now().toString().slice(-6));
  return await safeFetch(url.toString());
}

/* =================== Charts =================== */
function resetCharts() {
  if (state.priceChart)  { state.priceChart.remove();  state.priceChart=null; }
  if (state.volumeChart) { state.volumeChart.remove(); state.volumeChart=null; }
  state.candleSeries = null; state.volumeSeries = null;
  state._markersCache = [];
  clearTradeLines();
}

function buildPriceChart() {
  const el = document.getElementById('chartPrice');
  const chart = LightweightCharts.createChart(el, {
    width: el.clientWidth,
    height: el.clientHeight,
    layout: { background: { color: '#0f141b' }, textColor: '#E6F0FF' },
    grid: { vertLines: { color: '#2a3544' }, horzLines: { color: '#2a3544' } },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    rightPriceScale: { borderVisible: false },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
    handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: false },
    watermark: { visible: true, color: 'rgba(255,255,255,0.05)', text: 'ScalpTV', fontSize: 16,
                 horzAlign: 'left', vertAlign: 'bottom' }
  });

  const ts = chart.timeScale();
  ts.applyOptions({
    rightOffset: FUTURE_OFFSET_BARS,
    barSpacing: Math.max(4, Math.min(14, 8)),
    rightBarStaysOnScroll: true,
    borderVisible: false,
    secondsVisible: false,
    timeVisible: true,
    shiftVisibleRangeOnNewBar: true
  });

  el.addEventListener('dblclick', function(){
    state.followRealtime = true;
    try {
      ts.setRightOffset(FUTURE_OFFSET_BARS); ts.scrollToRealTime();
      if (state.volumeChart && state.volumeChart.timeScale) {
        var vts = state.volumeChart.timeScale(); if (vts) { vts.setRightOffset(FUTURE_OFFSET_BARS); vts.scrollToRealTime(); }
      }
    } catch (e) {}
    var btn = document.getElementById('toggleLive'); if (btn) btn.classList.add('active');
  });

  ts.subscribeVisibleTimeRangeChange(function(range){
    if (!range) return;
    var last = state.lastTs || 0;
    if (range.to && (range.to < (last - stepSec()))) {
      state.followRealtime = false;
      var btn = document.getElementById('toggleLive'); if (btn) btn.classList.remove('active');
    }
    reapplyMarkers(range);
  });

  ts.subscribeVisibleLogicalRangeChange(function(){
    try {
      var opts = ts.options();
      var bs = opts.barSpacing || 8;
      var clamped = Math.max(2, Math.min(30, bs));
      if (clamped !== bs) ts.applyOptions({ barSpacing: clamped });
    } catch (e) {}
  });

  const candleSeries = chart.addCandlestickSeries({
    upColor: '#4fff85', downColor: '#ff4976',
    borderUpColor: '#4fff85', borderDownColor: '#ff4976',
    wickUpColor: '#4fff85', wickDownColor: '#ff4976'
  });

  chart.subscribeClick(function(param){
    if (!param || !param.time) return;
    const t = (typeof param.time === 'number') ? param.time : param.time.timestamp;
    const idx = nearestTradeIndex(t);
    if (idx == null) return;
    drawTradeLines(state.signals[idx]);
  });

  el.addEventListener('contextmenu', function(ev){ ev.preventDefault(); clearTradeLines(); });
  ts.subscribeVisibleTimeRangeChange(function(range){ reapplyMarkers(range); });

  new ResizeObserver(function(es){ es.forEach(function(e){ chart.resize(e.contentRect.width, e.contentRect.height); }); }).observe(el);

  state.priceChart = chart;
  state.candleSeries = candleSeries;
}

function buildVolumeChart() {
  const el = document.getElementById('chartVolume');
  const chart = LightweightCharts.createChart(el, {
    width: el.clientWidth,
    height: el.clientHeight,
    layout: { background: { color: '#0f141b' }, textColor: '#E6F0FF' },
    grid: { vertLines: { color: '#2a3544' }, horzLines: { color: '#2a3544' } },
    rightPriceScale: { borderVisible: false },
    handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true },
    handleScale: { mouseWheel: true, pinch: true, axisPressedMouseMove: false }
  });

  const ts = chart.timeScale();
  ts.applyOptions({
    rightOffset: FUTURE_OFFSET_BARS,
    barSpacing: Math.max(4, Math.min(14, 8)),
    rightBarStaysOnScroll: true,
    borderVisible: false,
    secondsVisible: false,
    timeVisible: true,
    shiftVisibleRangeOnNewBar: true
  });

  ts.subscribeVisibleTimeRangeChange(function(range){ reapplyMarkers(range); });

  const volumeSeries = chart.addHistogramSeries({
    priceFormat: { type: 'volume' }, priceScaleId: '',
    scaleMargins: { top: 0.8, bottom: 0 }
  });

  new ResizeObserver(function(es){ es.forEach(function(e){ chart.resize(e.contentRect.width, e.contentRect.height); }); }).observe(el);

  state.volumeChart = chart;
  state.volumeSeries = volumeSeries;
}

function syncCharts() {
  if (!state.priceChart || !state.volumeChart) return;
  const tsP = state.priceChart.timeScale();
  const tsV = state.volumeChart.timeScale();
  var block=false;
  function link(src,dst){
    src.subscribeVisibleTimeRangeChange(function(range){
      if (block||!range) return; block=true; dst.setVisibleRange(range); block=false;
    });
  }
  link(tsP,tsV); link(tsV,tsP);

  try { tsP.setRightOffset(FUTURE_OFFSET_BARS); tsV.setRightOffset(FUTURE_OFFSET_BARS); tsP.scrollToRealTime(); tsV.scrollToRealTime(); } catch (e) {}
}

/* =================== Tooltips =================== */
var $tip = document.getElementById('tooltip');
function showTip(x,y,html){ $tip.innerHTML=html; $tip.style.left=x+"px"; $tip.style.top=y+"px"; $tip.classList.remove('hidden'); }
function hideTip(){ $tip.classList.add('hidden'); }

function attachTooltips() {
  if (!state.priceChart || !state.candleSeries) return;
  state.priceChart.subscribeCrosshairMove(function(param){
    if (!param || !param.time) { hideTip(); return; }
    const bar = param.seriesData.get(state.candleSeries); if (!bar) { hideTip(); return; }
    const fx = state.currency==='BRL'?state.fx:1;
    const tSec = typeof param.time==='number' ? param.time : param.time.timestamp;
    const o=fmt(bar.open,2), h=fmt(bar.high,2), l=fmt(bar.low,2), c=fmt(bar.close,2);

    const idx = nearestTradeIndex(tSec);
    let sigHtml = '';
    if (idx != null) {
      const s = state.signals[idx];
      const entry=isFiniteNumber(s.entry)?fmt(s.entry*fx,2):'-';
      const stop =isFiniteNumber(s.stop)?fmt(s.stop*fx,2):'-';
      const tgt  =isFiniteNumber(s.target)?fmt(s.target*fx,2):'-';
      const risk = (isFiniteNumber(s.entry)&&isFiniteNumber(s.stop)) ? Math.abs(s.entry - s.stop) : 0;
      const rr   = (risk && isFiniteNumber(s.target)) ? Math.abs(s.target - s.entry)/risk : 0;
      const out  = s._outcome ? '  <b>'+s._outcome+'</b>' : (s._open ? '  <b>OPEN</b>' : '');
      const pnl  = (s._pnl_usdt != null) ? '  P/L '+fmt(s._pnl_usdt,2)+' USDT' : '';
      sigHtml = '<div style="margin-top:6px">' +
        '<b>#'+s._id+'</b> ' + (s.dir>0?'LONG':'SHORT') + out + pnl + '<br/>' +
        'entry ' + entry + '  stop ' + stop + '  tgt ' + tgt + '<br/>' +
        '<span class="muted">R:R ' + fmt(rr||0,2) + '</span>' +
      '</div>';

      if (Array.isArray(s._events) && s._events.length) {
        const evs = s._events.map(function(ev){
          const when = new Date(ev.time*1000).toLocaleTimeString();
          const sh = (ev.share!=null) ? (fmt(ev.share*100,0)+'%') : '';
          return '<div class="muted">'+when+'  '+ev.type+' @ '+fmt(ev.price*fx,2)+' '+sh+'</div>';
        }).join('');
        sigHtml += '<div style="margin-top:4px">'+evs+'</div>';
      }
    }

    showTip(param.point.x, param.point.y,
      '<div><b>'+new Date(tSec*1000).toLocaleString()+'</b></div>' +
      '<div>O '+o+'  H '+h+'  L '+l+'  C '+c+'</div>'+sigHtml);
  });
}

/* =================== Mapping / Metricas =================== */
function nearestTradeIndex(tSec){
  const tol = Math.floor(stepSec()/2);
  let best=null, bestd=1e12;
  for (let i=0;i<state.signals.length;i++){
    const ts = (state.signals[i]._plotTime != null)
      ? state.signals[i]._plotTime
      : Math.floor((state.signals[i].time||state.signals[i].entry_time)/1000);
    const d = Math.abs(ts - tSec);
    if (d < bestd && d <= tol){ best=i; bestd=d; }
  }
  return best;
}

/* =================== Lógica de Execução =================== */
function evaluateOutcomes(candles, signals, timeoutCfg){
  const byTime = new Map(candles.map(c => [c.time, c]));
  const timesAsc = candles.map(c => c.time).sort((a,b)=>a-b);

  function snapAtOrAfter(tSec) {
    if (byTime.has(tSec)) return tSec;
    for (let k = 1; k <= 3; k++) {
      const t2 = tSec + k * stepSec();
      if (byTime.has(t2)) return t2;
    }
    for (let i = 0; i < timesAsc.length; i++) {
      if (timesAsc[i] >= tSec) return timesAsc[i];
    }
    return null;
  }

  // Timeout do método (JSON) se houver
  const mm = (timeoutCfg && timeoutCfg.max_hold_map) ? timeoutCfg.max_hold_map : state.timeouts;
  let holdBars = Number(mm[state.tf]);
  if (!Number.isFinite(holdBars)) holdBars = 30;
  if (holdBars <= 0) holdBars = Number.POSITIVE_INFINITY;
  else holdBars = Math.max(1, Math.floor(holdBars * (HOLD_SCALE || 1.0)));

  const sStep = stepSec();

  // Config do método selecionado
  const mcfg = getActiveMethodConfig();
  const useATRStop = !!mcfg.use_atr_stop;
  const useATRTp   = !!mcfg.use_atr_tp;
  const atrStopLen = Number(mcfg.atr_stop_len) || 14;
  const atrStopMul = Number(mcfg.atr_stop_mult) || 1.0;
  const atrTpLen   = Number(mcfg.atr_tp_len)   || atrStopLen;
  const atrTpMul   = Number(mcfg.atr_tp_mult)  || 1.0;

  const atrStopMap = useATRStop ? buildATRMap(candles, atrStopLen) : null;
  const atrTpMap   = useATRTp   ? (useATRStop && atrTpLen === atrStopLen ? atrStopMap : buildATRMap(candles, atrTpLen)) : null;

  // prioridade intrabar (quando SL e TP estão dentro da mesma barra)
  const intrabarPriority = (
      (mcfg && (mcfg.exit_priority || mcfg.intrabar_priority)) || 'tp-first'
  ).toLowerCase();

  let blockUntilSec = -Infinity;

  signals.forEach(function(s, i){
    s._id = i + 1;
    s._outcome = null; s._exit=null; s._exitTimeSec=null; s._pnl_usdt=null; s._skipped=false; s._open=false;
    s._events = [];
    s._realized = 0; s._qtyLeft = 1.0;

    const dir = (s.dir > 0) ? 1 : -1;
    const entry = +s.entry;

    // --- STOP ---
    let stop = isFiniteNumber(s.stop) ? +s.stop : null;
    if (!isFiniteNumber(stop)) {
      if (useATRStop) {
        const start0 = Math.floor(Number(s.time)/1000);
        const atr = atrAt(atrStopMap, start0);
        if (isFiniteNumber(atr)) {
          stop = dir > 0 ? (entry - atrStopMul * atr) : (entry + atrStopMul * atr);
        }
      } else if (isFiniteNumber(mcfg.sl_fixed_perc)) {
        stop = initialStop(entry, dir, { sl_fixed_perc: Number(mcfg.sl_fixed_perc) });
      }
    }

    // --- TARGET(s) ---
    let tps = [];
    let shares = [];
    if (isFiniteNumber(s.target)) {
      tps = [+s.target]; shares = [1.0];
    } else if (useATRTp) {
      const start0 = Math.floor(Number(s.time)/1000);
      const atr = atrAt(atrTpMap, start0);
      if (isFiniteNumber(atr)) {
        const px = dir > 0 ? (entry + atrTpMul * atr) : (entry - atrTpMul * atr);
        tps = [px]; shares=[1.0];
      }
    } else if (state.useProgTP) {
      tps = buildProgressiveTargets(entry, dir, PROG_TP_DEFAULTS);
      shares = (Array.isArray(PROG_TP_DEFAULTS.tp_shares) && PROG_TP_DEFAULTS.tp_shares.length === tps.length)
        ? PROG_TP_DEFAULTS.tp_shares.slice() : [0.4,0.4,0.2];
    }

    const start = Math.floor(Number(s.time) / 1000);
    const endWindowSec = start + holdBars * sStep;

    if (!ALLOW_OVERLAP && start <= blockUntilSec) { s._skipped = true; return; }

    for (let k = 1; k <= holdBars; k++) {
      const t = start + k * sStep;
      const tEff = snapAtOrAfter(t);
      const c = tEff ? byTime.get(tEff) : null;
      if (!c) continue;

      const slHit = isFiniteNumber(stop) && (dir > 0 ? c.low <= stop : c.high >= stop);
      let tpHitIdx = -1;
      for (let tpIdx=0; tpIdx<tps.length; tpIdx++){
        if (shares[tpIdx] <= 0) continue;
        const tpPx = tps[tpIdx];
        const hit = (dir>0) ? (c.high >= tpPx) : (c.low <= tpPx);
        if (hit) { tpHitIdx = tpIdx; break; }
      }

      // colisão intrabar (SL e TP dentro da mesma barra)
      if (slHit && tpHitIdx >= 0) {
        if (intrabarPriority === 'stop-first') {
          const qty = s._qtyLeft;
          if (qty > 0) {
            const pnlPart = calcPnLPart(entry, stop, dir, qty);
            s._events.push({ type:'SL', time:tEff, price:stop, share:qty, pnl:pnlPart });
            s._realized += pnlPart; s._qtyLeft = 0;
          }
          s._outcome='SL'; s._exit=stop; s._exitTimeSec=tEff;
          break;
        } else {
          // tp-first
          const part = Math.min(shares[tpHitIdx], s._qtyLeft);
          if (part > 0) {
            const tpPx = tps[tpHitIdx];
            const pnlPart = calcPnLPart(entry, tpPx, dir, part);
            s._events.push({ type:(tps.length>1?('TP'+(tpHitIdx+1)):'TP'), time:tEff, price:tpPx, share:part, pnl:pnlPart });
            s._realized += pnlPart; s._qtyLeft = Math.max(0, s._qtyLeft - part);
            shares[tpHitIdx] = 0;

            if (state.useProgTP && tps.length > 1) {
              const moved = moveStopOnTP(stop, entry, tps, tpHitIdx, PROG_TP_DEFAULTS.step_on_tp, dir);
              if (isFiniteNumber(moved)) {
                if (dir > 0) stop = Math.max(stop || -Infinity, moved);
                else         stop = Math.min(stop ||  Infinity, moved);
              }
            }
          }
          if (s._qtyLeft <= 1e-9) {
            s._outcome = (tps.length>1?('TP'+(tpHitIdx+1)):'TP');
            s._exit = tps[tpHitIdx]; s._exitTimeSec = tEff;
            break;
          }
          // ainda na mesma barra, checa SL para o restante
          if (isFiniteNumber(stop)) {
            const qty = s._qtyLeft;
            if ((dir>0 ? c.low <= stop : c.high >= stop) && qty > 0) {
              const pnlPart = calcPnLPart(entry, stop, dir, qty);
              s._events.push({ type:'SL', time:tEff, price:stop, share:qty, pnl:pnlPart });
              s._realized += pnlPart; s._qtyLeft = 0;
              s._outcome='SL'; s._exit=stop; s._exitTimeSec=tEff;
              break;
            }
          }
          continue;
        }
      }

      // 1) SL isolado
      if (slHit) {
        const qty = s._qtyLeft;
        if (qty > 0) {
          const pnlPart = calcPnLPart(entry, stop, dir, qty);
          s._events.push({ type:'SL', time:tEff, price:stop, share:qty, pnl:pnlPart });
          s._realized += pnlPart; s._qtyLeft = 0;
        }
        s._outcome='SL'; s._exit=stop; s._exitTimeSec=tEff;
        break;
      }

      // 2) TP isolado
      if (tpHitIdx >= 0) {
        const part = Math.min(shares[tpHitIdx], s._qtyLeft);
        if (part > 0) {
          const tpPx = tps[tpHitIdx];
          const pnlPart = calcPnLPart(entry, tpPx, dir, part);
          s._events.push({ type:(tps.length>1?('TP'+(tpHitIdx+1)):'TP'), time:tEff, price:tpPx, share:part, pnl:pnlPart });
          s._realized += pnlPart; s._qtyLeft = Math.max(0, s._qtyLeft - part);
          shares[tpHitIdx] = 0;

          if (state.useProgTP && tps.length > 1) {
            const moved = moveStopOnTP(stop, entry, tps, tpHitIdx, PROG_TP_DEFAULTS.step_on_tp, dir);
            if (isFiniteNumber(moved)) {
              if (dir > 0) stop = Math.max(stop || -Infinity, moved);
              else         stop = Math.min(stop ||  Infinity, moved);
            }
          }
        }
        if (s._qtyLeft <= 1e-9) {
          s._outcome = (tps.length>1?('TP'+(tpHitIdx+1)):'TP');
          s._exit = tps[tpHitIdx]; s._exitTimeSec = tEff;
          break;
        }
      }

      // 3) REV (flip)
      if (!s._outcome && !ALLOW_OVERLAP) {
        for (let j = i + 1; j < signals.length; j++) {
          const s2 = signals[j];
          const t2 = Math.floor(Number(s2.time) / 1000);
          if (t2 > endWindowSec) break;
          if (s2.dir && s2.dir !== s.dir) {
            const revPx = +s2.entry;
            const qty = s._qtyLeft;
            if (qty > 0) {
              const pnlPart = calcPnLPart(entry, revPx, dir, qty);
              const tEff2 = snapAtOrAfter(t2) || t2;
              s._events.push({ type:'REV', time:tEff2, price:revPx, share:qty, pnl:pnlPart });
              s._realized += pnlPart; s._qtyLeft = 0;
            }
            s._outcome='REV'; s._exit=revPx; s._exitTimeSec=(snapAtOrAfter(t2) || t2);
            break;
          }
        }
      }
      if (s._outcome) break;
    }

    // 4) Timeout
    if (!s._outcome) {
      const tEff = snapAtOrAfter(endWindowSec);
      if (tEff && byTime.get(tEff)) {
        const cEnd = byTime.get(tEff);
        const qty = s._qtyLeft;
        if (qty > 0) {
          const pnlPart = calcPnLPart(entry, cEnd.close, dir, qty);
          s._events.push({ type:'TO', time:tEff, price:cEnd.close, share:qty, pnl:pnlPart });
          s._realized += pnlPart; s._qtyLeft = 0;
        }
        s._outcome='TO'; s._exit=cEnd.close; s._exitTimeSec=tEff;
      } else {
        s._open = true;
      }
    }

    // PnL com taxas/slippage
    if (s._events.length > 0) {
      s._pnl_usdt = s._events.reduce((acc, ev) =>
        acc + applyFeesAndSlippage(entry, ev.price, dir, ev.share), 0);
      blockUntilSec = (!ALLOW_OVERLAP) ? (s._exitTimeSec != null ? s._exitTimeSec : endWindowSec) : -Infinity;
    } else {
      s._pnl_usdt = null;
      blockUntilSec = (!ALLOW_OVERLAP) ? endWindowSec : -Infinity;
    }
  });

  state.tradeIndexMap.clear();
  signals.forEach((s, i) => state.tradeIndexMap.set(toSec(s.time), i));

  function calcPnLPart(entry, price, dir, share) {
    const gross = (price - entry) * (dir > 0 ? 1 : -1);
    return gross * share;
  }
  function applyFeesAndSlippage(entry, price, dir, share) {
    const mult = notionalMult() || 0;
    const fee = Number.isFinite(state.feePerc) ? state.feePerc : FEE_PERC;
    const slp = Number.isFinite(state.slippagePerc) ? state.slippagePerc : SLIPPAGE_PERC;
    const slipIn  = entry * slp;
    const slipOut = price * slp;
    const effEntry = dir > 0 ? (entry + slipIn) : (entry - slipIn);
    const effExit  = dir > 0 ? (price - slipOut) : (price + slipOut);
    const feeIn  = Math.abs(effEntry) * fee * mult * share;
    const feeOut = Math.abs(effExit)  * fee * mult * share;
    const pnlPrice = (effExit - effEntry) * (dir > 0 ? 1 : -1) * share;
    return (pnlPrice * mult) - (feeIn + feeOut);
  }
}

/* =================== Linhas de preco =================== */
function clearTradeLines(){
  if (!state.candleSeries || !state.tradeLines.length) return;
  state.tradeLines.forEach(function(l){ try{ state.candleSeries.removePriceLine(l); }catch(e){} });
  state.tradeLines = [];
}

function drawTradeLines(s) {
  if (!state.candleSeries) return;
  clearTradeLines();
  const fx = state.currency==='BRL'?state.fx:1;
  function mk(price, color, title, lineStyle){
    return state.candleSeries.createPriceLine({ price: price*fx, color: color, lineWidth: 1, lineStyle: (lineStyle==null?2:lineStyle), axisLabelVisible: true, title: title });
  }

  if (isFiniteNumber(s.entry))  state.tradeLines.push(mk(+s.entry, '#7aa0ff', 'ENTRY'));
  if (isFiniteNumber(s.stop))   state.tradeLines.push(mk(+s.stop,  '#ff6161', 'STOP'));

  if (isFiniteNumber(s.target)) {
    state.tradeLines.push(mk(+s.target, '#6cff8f', 'TARGET'));
  } else if (state.useProgTP && isFiniteNumber(s.entry)) {
    const tps = buildProgressiveTargets(+s.entry, s.dir>0?1:-1, PROG_TP_DEFAULTS);
    if (tps[0] != null) state.tradeLines.push(mk(tps[0], '#6cff8f', 'TP1'));
    if (tps[1] != null) state.tradeLines.push(mk(tps[1], '#6cff8f', 'TP2'));
    if (tps[2] != null) state.tradeLines.push(mk(tps[2], '#6cff8f', 'TP3'));
  } else if (isFiniteNumber(s.entry) && isFiniteNumber(s.stop)) {
    const rr = state.rrDefault || 2.0;
    const dir = s.dir > 0 ? 1 : -1;
    const risk = Math.abs(+s.entry - +s.stop);
    const synth = dir > 0 ? (+s.entry + rr * risk) : (+s.entry - rr * risk);
    state.tradeLines.push(mk(synth, '#6cff8f', 'TARGET*', 1));
  }
}

/* =================== Markers =================== */
function staggerMarkersByTime(marks){
  if (!Array.isArray(marks) || marks.length === 0) return [];
  const byT = new Map();
  for (let i = 0; i < marks.length; i++) {
    const m = marks[i];
    if (!m || m.time == null) continue;
    const arr = byT.get(m.time) || [];
    arr.push(m);
    byT.set(m.time, arr);
  }
  const out = [];
  for (const kv of byT.entries()) {
    const t = kv[0], arr = kv[1];
    for (let idx = 0; idx < arr.length; idx++) {
      const m = arr[idx];
      const mm = {}; for (var k in m) { if (Object.prototype.hasOwnProperty.call(m,k)) mm[k]=m[k]; }

      if (idx % 2 === 1) {
        if (mm.position === 'aboveBar') mm.position = 'belowBar';
        else if (mm.position === 'belowBar') mm.position = 'aboveBar';
      }
      if (idx > 0 && typeof mm.text === 'string' && mm.text.length > 18) {
        const match = mm.text.match(/^#\d+/); mm.text = (match && match.length > 0) ? match[0] : '';
      }
      out.push(mm);
    }
  }
  return out;
}

function buildMarkersFromSignals(signals){
  const marks = [];
  // Entradas
  signals.forEach((s, idx) => {
    // tempo para plot
    let t = s._plotTime;
    if (t == null){
      let rawt = s.entry_time || s.time;
      if (rawt > 1e12) rawt = Math.floor(rawt/1000);
      t = rawt;
    }
    marks.push({
      time: t,
      position: (s.dir > 0 ? 'belowBar' : 'aboveBar'),
      color: (s.dir > 0 ? 'green' : 'red'),
      shape: (s.dir > 0 ? 'arrowUp' : 'arrowDown'),
      text: `#${idx+1} ENTRY ${isFiniteNumber(s.entry)?fmt(s.entry,2):''}`
    });

    // Saídas do backend (exit_time/exit_price) OU eventos simulados
    if (s.exit_time && s.exit_price) {
      const reason = String(s.exit_reason || s.reason || '').toLowerCase();
      let color = 'yellow', label = 'EXIT';
      if (reason === 'target') { color='green'; label='TP'; }
      else if (reason === 'stop') { color='red'; label='SL'; }
      else if (reason === 'timeout') { color='orange'; label='TIMEOUT'; }
      else if (reason === 'flip') { color='blue'; label='FLIP'; }

      marks.push({
        time: Math.floor(Number(s.exit_time)/1000),
        position: (s.dir > 0 ? 'aboveBar' : 'belowBar'),
        color: color,
        shape: 'circle',
        text: `#${idx+1} ${label} ${fmt(s.exit_price,2)}${(s._pnl_usdt!=null)?` (${s._pnl_usdt>=0?'+':''}${fmt(s._pnl_usdt,2)} USDT)`:''}`
      });
    } else if (Array.isArray(s._events)) {
      s._events.forEach(ev=>{
        let color = '#9fb0c8', shape='circle';
        if (ev.type==='SL') color='#ff4976';
        else if (/^TP\d?$/.test(ev.type)||ev.type==='TP') color='#4fff85';
        else if (ev.type==='TO') color='#f0ad4e';
        else if (ev.type==='REV') color='#9fb0c8';
        marks.push({
          time: Math.floor(Number(ev.time)),
          position: (s.dir > 0 ? 'aboveBar' : 'belowBar'),
          color, shape,
          text: `#${s._id} ${ev.type}${(ev.share!=null)?' '+fmt(ev.share*100,0)+'%':''}`
        });
      });
    }
  });

  const all = staggerMarkersByTime(marks);
  const clipped = all.length > MAX_MARKERS ? all.slice(-MAX_MARKERS) : all;
  return clipped;
}

function reapplyMarkers(){
  if (!state.candleSeries) return;
  const markers = state._markersCache || [];
  try {
    const ts = state.priceChart && state.priceChart.timeScale ? state.priceChart.timeScale() : null;
    if (!ts){ state.candleSeries.setMarkers(markers); return; }
    const r = ts.getVisibleRange && ts.getVisibleRange();
    if (r && Number.isFinite(r.from) && Number.isFinite(r.to)) {
      const pad = stepSec()*20;
      const from = Math.floor(r.from - pad);
      const to   = Math.floor(r.to   + pad);
      const filtered = markers.filter(m => m.time >= from && m.time <= to);
      state.candleSeries.setMarkers(filtered.length ? filtered : markers);
    } else {
      state.candleSeries.setMarkers(markers);
    }
  } catch (e) {}
}

/* =================== Data binding =================== */
async function setSeriesDataFromCandles(raw) {
  const fx = state.currency==='BRL'?state.fx:1;

  // 1) sanitiza + log
  const mappedRaw = sanitizeCandles(raw);
  if (mappedRaw.length === 0) {
    console.warn("[candles] vazio do backend; mantendo gráfico e tentando reload depois");
    return [];
  }

  // 2) normaliza
  const serie = mappedRaw.map(c => normalizeForSeries(c, fx)).filter(Boolean);

  // 3) fallback: poucos candles = força reload full da API (ignorando cache)
  if (serie.length < MIN_INIT_BARS) {
    console.warn(`[candles] só ${serie.length} candles após normalizar; forçando reload full`);
    const fresh = await fetchCandles(candlesLimitForTF(state.tf));
    const freshMapped = sanitizeCandles(fresh);
    const freshSerie = freshMapped.map(c => normalizeForSeries(c, fx)).filter(Boolean);
    if (freshSerie.length >= MIN_INIT_BARS) {
      if (!state.candleSeries || !state.volumeSeries) return freshMapped;
      state.candleSeries.setData(freshSerie.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));
      state.volumeSeries.setData(freshSerie.map(c => ({
        time: c.time, value: Number.isFinite(c.volume) ? c.volume : 0,
        color: (c.baseClose >= c.baseOpen ? 'rgba(76,175,80,.6)' : 'rgba(244,67,54,.6)')
      })));
      state.lastTs = freshSerie[freshSerie.length - 1].time;
      return freshMapped;
    }
  }

  // 4) aplica no gráfico (normal)
  if (!state.candleSeries || !state.volumeSeries) return mappedRaw;
  try {
    state.candleSeries.setData(serie.map(c => ({ time: c.time, open: c.open, high: c.high, low: c.low, close: c.close })));
    state.volumeSeries.setData(serie.map(c => ({
      time: c.time,
      value: Number.isFinite(c.volume) ? c.volume : 0,
      color: (c.baseClose >= c.baseOpen ? 'rgba(76,175,80,.6)' : 'rgba(244,67,54,.6)')
    })));
    state.lastTs = (serie.length ? serie[serie.length - 1].time : 0);
  } catch (e) {
    console.error("setData error", e);
  }

  // 5) mantém live-on e ajusta offset
  try {
    const tsP = state.priceChart.timeScale();
    const tsV = state.volumeChart.timeScale();
    tsP.setRightOffset(FUTURE_OFFSET_BARS);
    tsV.setRightOffset(FUTURE_OFFSET_BARS);
    tsP.scrollToRealTime();
    tsV.scrollToRealTime();
    state.followRealtime = true;
    var btn = document.getElementById('toggleLive'); if (btn) btn.classList.add('active');
  } catch (e) {}

  return mappedRaw;
}

async function refreshSignalsAndOutcomes(candles) {
  const sigResp = await fetchSignalsFor(state.methodId);

  // Preferir executions (já vêm com exits e pnl do backend)
  let signals = [];
if (sigResp.executions && sigResp.executions.length) {
  signals = sigResp.executions;
} else if (sigResp.signals && sigResp.signals.length) {
  signals = sigResp.signals;
} else {
  signals = state.signals; // mantém últimos sinais se backend devolveu vazio
}

  state.signals = signals.sort((a,b)=> (a.time||a.entry_time) - (b.time||b.entry_time));

  // Normalizar tempo para segundos + criar _id e _plotTime (desloca quando há empates)
  const byTime = {};
  state.signals.forEach((s, i) => {
    let t = s.entry_time || s.time;
    if (t > 1e12) t = Math.floor(t/1000);
    if (!byTime[t]) byTime[t] = 0;
    const offset = byTime[t]++;
    s._plotTime = t + offset; // desloca 1s por sinal extra no mesmo candle
    s._id = i+1;
  });

  // Se o backend não devolveu exits/pnl, avalia localmente com as regras do método
  const needsLocalEval = !state.signals.some(s =>
    s._pnl_usdt != null || (s.exit_time && s.exit_price) || (Array.isArray(s._events) && s._events.length)
  );
  if (needsLocalEval && Array.isArray(candles) && candles.length) {
    const timeoutCfg = buildTimeoutCfgFromMethod();  // lê max_hold do JSON
    evaluateOutcomes(candles, state.signals, timeoutCfg);
  }

  // Markers
  if (state.signals.length > 0) {
  state._markersCache = buildMarkersFromSignals(state.signals);
}
reapplyMarkers();
console.log("[ML DEBUG] plotando", state._markersCache.length, "markers a partir de", state.signals.length, "sinais");

  // 📊 Atualiza header (PnL da janela, ex.: 720h)
  const cutoff = Date.now() - (state.hoursLookback * 60 * 60 * 1000);

  function tradeTimeMs(s) {
    if (s.exit_time != null) {
      // suporta string ISO, segundos ou ms
      if (typeof s.exit_time === "string") return new Date(s.exit_time).getTime();
      const et = Number(s.exit_time);
      return et > 1e12 ? et : et * 1000;
    }
    if (s._exitTimeSec != null) return s._exitTimeSec * 1000;
    if (s.time != null) {
      const t = Number(s.time);
      return t > 1e12 ? t : t * 1000;
    }
    return null;
  }

  const recent = state.signals.filter(s => {
    const t = tradeTimeMs(s);
    return t == null || t >= cutoff;   // inclui abertos
  });

  const pnlAgg = recent.reduce((acc, s) => {
  if (s._pnl_usdt != null) return acc + s._pnl_usdt;
  if (s.pnl != null) return acc + s.pnl;   // fallback para backend antigo
  return acc;
}, 0);

  const nExec  = recent.length;

  const pillPNL = document.getElementById('pillPNL');
  const pillN   = document.getElementById('pillN');

  if (pillPNL) {
    pillPNL.textContent = "P/L " + state.hoursLookback + "h: " +
      (pnlAgg >= 0 ? '+' : '') + fmt(pnlAgg, 2) + " USDT";
    if (pnlAgg >= 0) { 
      pillPNL.classList.add('gain'); 
      pillPNL.classList.remove('loss'); 
    } else { 
      pillPNL.classList.remove('gain'); 
      pillPNL.classList.add('loss'); 
    }
  }
  if (pillN) { 
    pillN.textContent = "Trades: " + nExec; 
  }
}   // <<< aqui fecha a função corretamente

/* =================== Metodos/Combos =================== */
function fillMethodsDropdown(list){
  const sel = document.getElementById('selMethod');
  sel.innerHTML='';
  state.methodMap.clear();

  list.forEach(function(obj, idx){
    if (!obj) return;
    // 'id' pode não vir no seu JSON; gerar a partir de 'method' ou do índice
    const id = obj.id || obj.method || obj.name || ('m'+idx);
    const label = obj.label || obj.method || obj.name || id;
    const full = { ...obj, id, label };
    state.methodMap.set(id, full);
    const opt = document.createElement('option');
    opt.value = id;
    opt.textContent = label;
    sel.appendChild(opt);
  });

  if (!state.methodId || !state.methodMap.has(state.methodId)) {
    const first = list[0];
    state.methodId = first ? (first.id || first.method || first.name || 'm0') : null;
  }
  sel.value = state.methodId || '';
}

async function loadAndFillMethods(){
  const resp = await fetchMethods();

  // resp pode ter formato { timeframes: { '1m': [...], '5m': [...] } } ou lista direta
  let arr = [];
  if (resp && resp.timeframes && Array.isArray(resp.timeframes[state.tf])) {
    arr = resp.timeframes[state.tf];
  } else if (Array.isArray(resp)) {
    arr = resp;
  } else if (resp && Array.isArray(resp.methods)) {
    arr = resp.methods;
  }
  fillMethodsDropdown(Array.isArray(arr)?arr:[]);
}

/* =================== Primeiro carregamento =================== */
async function firstLoad(rebuildCharts){
  if (rebuildCharts == null) rebuildCharts = true;
  await fetchFX();
  if (rebuildCharts) { resetCharts(); buildPriceChart(); buildVolumeChart(); }

  const cache = loadCache();
  let candles;
  const need = candlesLimitForTF(state.tf);
  if (cache && cache.candles && cache.tf===state.tf && Array.isArray(cache.candles) && cache.candles.length>=need) {
    candles = await setSeriesDataFromCandles(cache.candles);
  } else {
    const raw = await fetchCandles(need);
    candles = await setSeriesDataFromCandles(raw);
    // só salva cache se veio quantidade decente de barras
    if (Array.isArray(raw) && raw.length >= MIN_INIT_BARS) {
      saveCache({ tf: state.tf, methodId: state.methodId, viewType: state.viewType, candles: raw });
    }
  }
  state.baseCandles = candles.slice();

  await loadAndFillMethods();
  await refreshSignalsAndOutcomes(candles);

  syncCharts(); attachTooltips();

  if (state.priceChart) {
    state.priceChart.timeScale().subscribeVisibleTimeRangeChange(function(){ reapplyMarkers(); });
  }
}

/* =================== Incremental =================== */
let _lastSignalsPoll = 0;
async function loadCandlesIncremental() {
  if (!state.candleSeries || !state.volumeSeries) return;

  const fx = state.currency==='BRL'?state.fx:1;
  const raw = await fetchCandles(2, state.lastTs);

  // sanitiza e pega só novos
  const mapped = sanitizeCandles(raw)
    .filter(c => Number(c.time) >= (state.lastTs || 0))
    .sort((a, b) => a.time - b.time);

  // se não veio nada novo e a série está "stale" por tempo demais -> reload full
  if (mapped.length === 0) {
    const lagSec = nowSec() - (state.lastTs || 0);
    if (lagSec > STALE_RELOAD_MULT * stepSec()) {
      console.warn(`[incremental] sem barras novas por ${lagSec}s; forçando reload`);
      const base = sanitizeCandles(await fetchCandles(candlesLimitForTF(state.tf)));
      await setSeriesDataFromCandles(base);
      await refreshSignalsAndOutcomes(base);
    }
    return;
  }

  let appendedNewBar = false;
  for (let i = 0; i < mapped.length; i++) {
    const sc = normalizeForSeries(mapped[i], fx);
    if (!sc) continue;
    appendedNewBar = appendedNewBar || (sc.time > state.lastTs);
    try {
      state.candleSeries.update({ time: sc.time, open: sc.open, high: sc.high, low: sc.low, close: sc.close });
      state.volumeSeries.update({ time: sc.time, value: Number.isFinite(sc.volume) ? sc.volume : 0, color: (sc.baseClose >= sc.baseOpen ? 'rgba(76,175,80,.6)' : 'rgba(244,67,54,.6)') });
      state.lastTs = sc.time;
    } catch (e) {
      console.error("update error", e, sc);
    }
  }

  if (appendedNewBar && state.followRealtime) {
    try {
      const tsP = state.priceChart.timeScale();
      const tsV = state.volumeChart.timeScale();
      tsP.setRightOffset(FUTURE_OFFSET_BARS);
      tsV.setRightOffset(FUTURE_OFFSET_BARS);
      tsP.scrollToRealTime();
      tsV.scrollToRealTime();
    } catch (e) {}
  }

  reapplyMarkers();

  // repuxa sinais periodicamente
  const now = Date.now();
  if (now - _lastSignalsPoll > POLL_MS_SIGNALS) {
    _lastSignalsPoll = now;
    const base = sanitizeCandles(await fetchCandles(candlesLimitForTF(state.tf)));
    state.baseCandles = base.slice();
    await refreshSignalsAndOutcomes(base);
  }
}

/* =================== UI =================== */
function bindTfButtons(){
  const btns = document.querySelectorAll("#tfGroup button[data-tf]");
  for (let i=0;i<btns.length;i++){
    const btn = btns[i];
    btn.addEventListener("click", async function(){
      const all = document.querySelectorAll("#tfGroup button");
      for (let j=0;j<all.length;j++) all[j].classList.remove("active");
      btn.classList.add("active");
      state.tf = btn.dataset.tf;
      state.methodId = null;
      await firstLoad(true);
    });
  }
}
function bindCurrencyButtons(){
  const usd = document.getElementById("curUSD");
  const brl = document.getElementById("curBRL");
  usd.addEventListener("click", async function(){
    usd.classList.add("active"); brl.classList.remove("active");
    state.currency="USD"; await firstLoad(false);
  });
  brl.addEventListener("click", async function(){
    brl.classList.add("active"); usd.classList.remove("active");
    state.currency="BRL"; await firstLoad(false);
  });
}
function bindViewButtons(){
  const bBase  = document.getElementById("btnBase"),
        bCombo = document.getElementById("btnCombo"),
        bML    = document.getElementById("btnML");   // novo botão ML

  bBase.addEventListener("click", async function(){
    bBase.classList.add('active'); bCombo.classList.remove('active'); if (bML) bML.classList.remove('active');
    state.viewType='base'; state.methodId=null; await firstLoad(false);
  });
  bCombo.addEventListener("click", async function(){
    bCombo.classList.add('active'); bBase.classList.remove('active'); if (bML) bML.classList.remove('active');
    state.viewType='combo'; state.methodId=null; await firstLoad(false);
  });
  if (bML){
    bML.addEventListener("click", async function(){
      bML.classList.add('active'); bBase.classList.remove('active'); bCombo.classList.remove('active');
      state.viewType='ml'; state.methodId=null; await firstLoad(false);
    });
  }
}
function bindMethodSelect(){
  const sel=document.getElementById("selMethod");
  sel.addEventListener("change", async function(e){
    state.methodId = e.target.value || null;
    const base = sanitizeCandles(await fetchCandles(candlesLimitForTF(state.tf)));
    state.baseCandles = base.slice();
    await refreshSignalsAndOutcomes(base);
  });
}
function bindLiveButton(){
  const btnLive = document.getElementById('toggleLive');
  if (!btnLive) return;
  btnLive.addEventListener('click', function(){
    state.followRealtime = !state.followRealtime;
    btnLive.classList.toggle('active', state.followRealtime);
    if (state.followRealtime) {
      try {
        const tsP = state.priceChart.timeScale();
        const tsV = state.volumeChart.timeScale();
        tsP.setRightOffset(FUTURE_OFFSET_BARS);
        tsV.setRightOffset(FUTURE_OFFSET_BARS);
        tsP.scrollToRealTime();
        tsV.scrollToRealTime();
      } catch (e) {}
    }
  });
}

/* Cria/Anexa controles no header para Contracts / Contract Value / TP Prog */
function ensureExecControlsUI(){
  let host = document.getElementById('headerControls')
         || document.getElementById('headerRight')
         || document.getElementById('header')
         || document.querySelector('.topbar')
         || document.body;

  let box = document.getElementById('execControls');
  if (!box){
    box = document.createElement('div');
    box.id = 'execControls';
    box.style.display = 'flex';
    box.style.gap = '8px';
    box.style.alignItems = 'center';
    box.style.marginLeft = '12px';
    box.style.flexWrap = 'wrap';
    box.innerHTML = `
      <label style="font-size:12px;color:#9fb0c8;">Qty
        <input id="inpContracts" type="number" min="0" step="0.01" style="width:80px;margin-left:4px;">
      </label>
      <label style="font-size:12px;color:#9fb0c8;">CV (BTC)
        <input id="inpContractValue" type="number" min="0" step="0.001" style="width:90px;margin-left:4px;">
      </label>
      <label style="font-size:12px;color:#9fb0c8;">
        <input id="chkTPProg" type="checkbox" style="vertical-align:middle;margin-right:4px;">
        TP Prog
      </label>
    `;
    host.appendChild(box);
  }

  const $qty = document.getElementById('inpContracts');
  const $cv  = document.getElementById('inpContractValue');
  const $tp  = document.getElementById('chkTPProg');
  $qty.value = String(state.contracts);
  $cv.value  = String(state.contractValue);
  $tp.checked = !!state.useProgTP;

  function onChangeExec(){
    const q = Number($qty.value);
    const cv = Number($cv.value);
    state.contracts = isFiniteNumber(q) ? q : state.contracts;
    state.contractValue = isFiniteNumber(cv) ? cv : state.contractValue;
    state.useProgTP = !!$tp.checked;
    saveExecSettings();
    if (Array.isArray(state.baseCandles) && state.baseCandles.length){
      refreshSignalsAndOutcomes(state.baseCandles);
      reapplyMarkers();
    }
  }
  $qty.addEventListener('change', onChangeExec);
  $cv.addEventListener('change', onChangeExec);
  $tp.addEventListener('change', onChangeExec);
}

function bindUI(){
  const reloadBtn = document.getElementById("reload");
  if (reloadBtn) reloadBtn.addEventListener("click", async function(){ await firstLoad(false); });
  bindTfButtons(); bindCurrencyButtons(); bindViewButtons(); bindMethodSelect(); bindLiveButton();
  ensureExecControlsUI();
}

/* =================== Bootstrap =================== */
document.addEventListener('DOMContentLoaded', async function () {
  try {
    loadExecSettings();
    bindUI();
    await firstLoad(true);
    setInterval(loadCandlesIncremental, POLL_MS_CANDLES);
  } catch (err) { console.error("Init error:", err); }
});
