/* Scalp TV Frontend - Lightweight Charts + API (robust loader) */
(() => {
  "use strict";
  const $ = (sel, root=document) => root.querySelector(sel);

  async function ensureLightweightCharts() {
    if (window.LightweightCharts && typeof LightweightCharts.createChart === 'function') return;
    // tenta local (caso index não tenha conseguido)
    await new Promise((resolve) => {
      const ok = () => resolve();
      const err = () => resolve(); // resolvemos de qualquer forma; se não carregar, vai falhar adiante
      const s = document.createElement('script');
      s.src = '/static/lightweight-charts.standalone.production.js';
      s.onload = ok; s.onerror = () => {
        const c = document.createElement('script');
        c.src = 'https://unpkg.com/lightweight-charts@4/dist/lightweight-charts.standalone.production.js';
        c.onload = ok; c.onerror = err; document.head.appendChild(c);
      };
      document.head.appendChild(s);
    });
  }

  const state = {
    symbol: 'BTCUSDT', tf: '5m',
    chart: null, candleSeries: null, volumeSeries: null,
    methods: [], methodId: null, lastSignals: null
  };

  const chartEl = $('#chart');
  const methodSel = $('#methodSelect');
  const fxEl = $('#fxpill');
  const pnlEl = $('#pnl48pill');
  const srcBadge = $('#srcBadge');

  function initChart() {
    if (!window.LightweightCharts || typeof LightweightCharts.createChart !== 'function') {
      console.error('LightweightCharts não carregada.');
      return;
    }
    if (!chartEl) return;

    const chart = LightweightCharts.createChart(chartEl, {
      width: chartEl.clientWidth,
      height: chartEl.clientHeight,
      layout: { background: { color: '#0c0c10' }, textColor: '#d0d0d0' },
      grid: { vertLines: { visible: false }, horzLines: { visible: false } },
      timeScale: { timeVisible: true, secondsVisible: false },
      rightPriceScale: { borderVisible: false },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal }
    });

    // proteção caso a API seja alterada
    if (typeof chart.addCandlestickSeries !== 'function') {
      console.error('API inesperada do LightweightCharts (sem addCandlestickSeries).');
      return;
    }

    const candleSeries = chart.addCandlestickSeries();
    const volumeSeries = chart.addHistogramSeries({ priceScaleId: 'left' });
    state.chart = chart;
    state.candleSeries = candleSeries;
    state.volumeSeries = volumeSeries;

    window.addEventListener('resize', () => {
      chart.applyOptions({ width: chartEl.clientWidth, height: chartEl.clientHeight });
    });
  }

  function setCandles(rows) {
    if (!state.candleSeries || !state.volumeSeries) return;
    const candles = rows.map(r => ({
      time: Math.floor(r.time / 1000),
      open: +r.open, high: +r.high, low: +r.low, close: +r.close
    }));
    const volume = rows.map(r => ({ time: Math.floor(r.time / 1000), value: +r.volume }));
    state.candleSeries.setData(candles);
    state.volumeSeries.setData(volume);
  }

  function markersFromPayload(payload) {
    const markers = [];
    for (const e of (payload.entries || [])) {
      markers.push({ id:`E:${e.time}`, time: Math.floor(e.time/1000), position:'belowBar', shape:'arrowUp', text:'LONG' });
    }
    for (const x of (payload.exits || [])) {
      markers.push({ id:`X:${x.time}`, time: Math.floor(x.time/1000), position:'aboveBar', shape:'arrowDown', text:(x.reason || 'EXIT') });
    }
    return markers;
  }
  function setMarkers(markers) {
    if (!state.candleSeries) return;
    const MAX_WITH_LABELS = 150, MAX_ONLY_ENTRIES = 350;
    let arr = [...markers];
    if (arr.length > MAX_WITH_LABELS) arr = arr.map(m => ({ ...m, text: undefined }));
    if (arr.length > MAX_ONLY_ENTRIES) {
      const entries = arr.filter(m => (m.id || '').startsWith('E'));
      const X = Math.ceil(entries.length / MAX_ONLY_ENTRIES);
      arr = entries.filter((_, i) => i % X === 0);
    }
    state.candleSeries.setMarkers(arr);
  }

  async function api(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return await r.json();
  }
  async function loadFX() {
    try {
      const j = await api(`/api/fx?pair=USDTBRL`);
      const px = j && typeof j.price === 'number' ? j.price : null;
      if (fxEl) fxEl.textContent = `USDTBRL ${px ? px.toFixed(2) : '—'}`;
    } catch { if (fxEl) fxEl.textContent = `USDTBRL —`; }
  }
  async function loadCandles() {
    const rows = await api(`/api/candles?symbol=${state.symbol}&tf=${state.tf}&limit=1200`);
    setCandles(rows);
  }
  async function loadMethods() {
    const rows = await api(`/api/methods?tf=${state.tf}`);
    rows.sort((a, b) => ((b.n_trades_48h ?? 0) - (a.n_trades_48h ?? 0)) ||
                         ((b.n_trades ?? 0) - (a.n_trades ?? 0)) ||
                         ((b.score ?? -1e18) - (a.score ?? -1e18)));
    state.methods = rows;
    methodSel.innerHTML = '';
    for (const r of rows) {
      const opt = document.createElement('option');
      opt.value = r.id;
      opt.textContent = `${r.label}  —  score: ${r.score ?? '—'}  | trades: ${r.n_trades ?? '—'}  | 48h: ${r.n_trades_48h ?? 0}`;
      methodSel.appendChild(opt);
    }
    if (rows.length) { methodSel.value = rows[0].id; state.methodId = rows[0].id; }
  }
  function isComboId(id) { return id && id.startsWith('COMBO:'); }

  async function loadSignals() {
    const id = state.methodId || '';
    const isCombo = isComboId(id);
    const now = Date.now(), from = now - 48*3600*1000;
    const url = new URL('/api/signals', location.origin);
    url.searchParams.set('symbol', state.symbol);
    url.searchParams.set('tf', state.tf);
    url.searchParams.set('from', String(from));
    url.searchParams.set('to', String(now));
    url.searchParams.set('full', '1');
    if (isCombo) url.searchParams.set('method_id', id);
    else url.searchParams.set('method', id.split('|')[0]);

    const payload = await api(url.toString());
    if (pnlEl && typeof payload.pnl_48h_usdt === 'number') {
      const v = payload.pnl_48h_usdt;
      pnlEl.textContent = `48h: ${v.toFixed(2)} USDT`;
      pnlEl.classList.toggle('gain', v >= 0);
      pnlEl.classList.toggle('loss', v < 0);
    }
    if (srcBadge) {
      srcBadge.textContent = isCombo ? 'LIVE' : 'CSV';
      srcBadge.classList.toggle('gain', isCombo);
      srcBadge.classList.toggle('loss', !isCombo);
    }
    setMarkers(markersFromPayload(payload));
  }

  document.addEventListener('DOMContentLoaded', async () => {
    await ensureLightweightCharts();
    initChart();
    await loadFX();
    await loadCandles();
    await loadMethods();
    await loadSignals();
    setInterval(loadFX, 30_000);
    setInterval(loadCandles, 3_000);
    setInterval(loadSignals, 10_000);
  });

  $('#methodSelect')?.addEventListener('change', async (e) => {
    state.methodId = e.target.value;
    await loadSignals();
  });
})();
