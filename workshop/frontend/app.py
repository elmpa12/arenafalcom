/* ======================================================================
   Scalp TV - app.js  (build 2025-09-12-1)
   - 48h sim + live
   - cores por trade, tooltips, SL/TP/TP2 ao clicar
   - métodos ordenados por nº de trades (48h) + score
   - correções de tempo (ms -> s) para markers/candles
   ====================================================================== */

(() => {
  // ------------------------- Helpers -------------------------
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));
  const nowMs = () => Date.now();
  const sec = (ms) => Math.floor(Number(ms) / 1000);
  const ms = (s) => Math.floor(Number(s) * 1000);

  const toSec = (t) => {
    if (t == null) return undefined;
    if (typeof t === 'number') return t > 1e12 ? Math.floor(t / 1000) : Math.floor(t);
    if (typeof t === 'string') {
      const d = Date.parse(t);
      return isNaN(d) ? undefined : Math.floor(d / 1000);
    }
    if (t instanceof Date) return Math.floor(t.getTime() / 1000);
    return undefined;
  };

  const tfMinutes = (tf) => {
    if (!tf) return 1;
    const m = String(tf).match(/^(\d+)(m|h|d)$/);
    if (!m) return 1;
    const n = Number(m[1]);
    const u = m[2];
    if (u === 'm') return n;
    if (u === 'h') return n * 60;
    if (u === 'd') return n * 60 * 24;
    return 1;
  };
  const tfSec = (tf) => tfMinutes(tf) * 60;

  const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

  const fmt = {
    pct(x)  { if (x == null || !isFinite(x)) return '0.00%'; return (x >= 0 ? '+' : '') + (x * 100).toFixed(2) + '%'; },
    brl(x)  { if (x == null || !isFinite(x)) return 'R$ 0,00'; return x.toLocaleString('pt-BR', { style: 'currency', currency: 'BRL', maximumFractionDigits: 2 }); },
    usd(x)  { if (x == null || !isFinite(x)) return 'US$ 0.00'; return x.toLocaleString('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 2 }); },
    num(x,n=2){ if (x==null || !isFinite(x)) return '0'; return Number(x).toFixed(n); },
    dt(s)   { return new Date(ms(s)).toISOString().replace('T',' ').slice(0,19) + 'Z'; }
  };

  const API = {
    candles: '/api/candles',
    methods: '/api/methods',
    signals: '/api/signals',
    fx: '/api/fx',
    health: '/api/health'
  };

  async function jget(url) {
    const r = await fetch(url, { cache: 'no-cache' });
    if (!r.ok) throw new Error(`${url} -> ${r.status}`);
    return r.json();
  }

  // ------------------------- Estado -------------------------
  const STATE = {
    symbol: localStorage.getItem('tv_symbol') || 'BTCUSDT',
    tf:     localStorage.getItem('tv_tf') || '5m',
    methodId: localStorage.getItem('tv_method') || '',
    dir:    localStorage.getItem('tv_dir') || 'both',
    currency: localStorage.getItem('tv_currency') || 'BRL',
    fx: { USDTBRL: 5.40 },
    // dados
    candles: [],            // {time(s), open, high, low, close, volume}
    vol: [],                // histogram data for volume chart
    methods: [],            // preenchido pelo /api/methods (score & trades48 se disponível)
    signals: null,          // payload bruto de /api/signals
    markers: [],            // markers construídos
    markersIndex: new Map(),// time(sec) -> array markers
    segments: [],           // trades/segments
    tradeColors: new Map(), // trade_key -> color
    selectedTradeKey: null,
    priceLines: [],         // price lines atuais no chart
    bootDone: false
  };

  // paleta (muitas cores, cicla)
  const COLOR_SET = [
    '#19B394','#1E90FF','#FFD166','#EF476F','#06D6A0','#F78C6B','#73D2DE','#E76F51',
    '#8338EC','#3A86FF','#FF006E','#FFBE0B','#8AC926','#1982C4','#6A4C93','#FF7F11',
    '#00A878','#F45B69','#E4FF1A','#118AB2','#073B4C','#FFD700','#90BE6D','#FB5607'
  ];

  const ATR_FACT = { '1m': {sl:1.1,tp1:1.0,tp2:1.8}, '5m': {sl:1.2,tp1:1.0,tp2:2.0}, '15m': {sl:1.3,tp1:1.0,tp2:2.2} };

  // ------------------------- Chart -------------------------
  let chart, volChart, series, volSeries;
  const chartEl   = $('#chart');
  const volEl     = $('#volChart');
  const tooltipEl = $('#tooltip');  // criado no index; overlay absoluto
  const plBadgeEl = $('#pl48h');

  function initCharts() {
    // protege contra recriação
    if (chart) { chart.remove(); chart = null; }
    if (volChart) { volChart.remove(); volChart = null; }

    const width  = chartEl.clientWidth;
    const height = chartEl.clientHeight;

    chart = LightweightCharts.createChart(chartEl, {
      width, height,
      layout: { background: { color: '#0f141b' }, textColor: '#C7CCD1' },
      grid: { vertLines: { color: '#18222D' }, horzLines: { color: '#18222D' } },
      rightPriceScale: { scaleMargins: { top: 0.06, bottom: 0.24 } },
      timeScale: { borderColor: '#18222D', timeVisible: true, secondsVisible: false }
    });
    series = chart.addCandlestickSeries({ upColor:'#26a69a', downColor:'#ef5350', borderUpColor:'#26a69a', borderDownColor:'#ef5350', wickUpColor:'#26a69a', wickDownColor:'#ef5350' });

    if (volEl) {
      const vH = Math.max(120, Math.round(window.innerHeight * 0.20));
      volChart = LightweightCharts.createChart(volEl, {
        width, height: vH,
        layout: { background: { color: '#0f141b' }, textColor: '#C7CCD1' },
        grid: { vertLines: { color: '#18222D' }, horzLines: { color: '#18222D' } },
        timeScale: { borderColor: '#18222D', timeVisible: true, secondsVisible: false }
      });
      volSeries = volChart.addHistogramSeries({ color: '#334155' });

      // sincroniza timeScale
      const sync = (src, dst) => src.timeScale().subscribeVisibleTimeRangeChange((r) => {
        const dr = dst.timeScale().getVisibleRange();
        if (!r || !dr || r.from !== dr.from || r.to !== dr.to) dst.timeScale().setVisibleRange(r);
      });
      sync(chart, volChart); sync(volChart, chart);
    }

    // hover tooltip
    chart.subscribeCrosshairMove(param => {
      if (!param || param.time === undefined) { tooltipEl.style.display = 'none'; return; }
      const t = Number(param.time);
      const markersAtT = STATE.markersIndex.get(t) || [];
      // monta tooltip preferindo marker mais “próximo” da posição do mouse em Y
      let best = null, dyBest = Infinity;
      for (const m of markersAtT) {
        if (!m.__seg) continue;
        const py = param?.seriesPrices?.get?.(series);
        if (py == null) { best = m; break; }
        const dy = Math.abs((m.__seg.entry_price || m.price || 0) - py);
        if (dy < dyBest) { dyBest = dy; best = m; }
      }
      if (!best) { tooltipEl.style.display = 'none'; return; }

      const seg = best.__seg;
      const isClosed = !!seg.exit_time;
      const side = (seg.side || seg.direction || '').toUpperCase();
      const pnlUsd = Number(seg.pnl_usd ?? seg.pnl ?? 0);
      const rate = STATE.currency === 'BRL' ? (STATE.fx.USDTBRL || 1) : 1;
      const pnl = pnlUsd * rate;
      const title = isClosed ? 'FECHADO' : 'ABERTO';
      const exr = (seg.exit_reason || seg.reason || '').toUpperCase();

      tooltipEl.innerHTML = `
        <div class="tt-title">${title} • ${side} • #${seg.trade_no ?? seg.id ?? '?'}</div>
        <div class="tt-row"><span>Entrada</span><b>${fmt.num(seg.entry_price)}</b></div>
        ${seg.exit_price ? `<div class="tt-row"><span>Saída</span><b>${fmt.num(seg.exit_price)}</b></div>`:''}
        ${seg.sl_price ? `<div class="tt-row"><span>Stop</span><b>${fmt.num(seg.sl_price)}</b></div>`:''}
        ${seg.tp1_price ? `<div class="tt-row"><span>TP1</span><b>${fmt.num(seg.tp1_price)}</b></div>`:''}
        ${seg.tp2_price ? `<div class="tt-row"><span>TP2</span><b>${fmt.num(seg.tp2_price)}</b></div>`:''}
        ${isClosed ? `<div class="tt-row"><span>Motivo</span><b>${exr || '-'}</b></div>`:''}
        <div class="tt-row"><span>P/L 48h</span><b>${STATE.currency==='BRL' ? fmt.brl(pnl) : fmt.usd(pnlUsd)}</b></div>
      `;
      // posiciona
      const box = chartEl.getBoundingClientRect();
      const x = clamp(param.point.x + 12, 8, box.width - tooltipEl.offsetWidth - 8);
      const y = clamp(param.point.y - tooltipEl.offsetHeight - 12, 8, box.height - tooltipEl.offsetHeight - 8);
      tooltipEl.style.left = x + 'px';
      tooltipEl.style.top  = y + 'px';
      tooltipEl.style.display = 'block';
    });

    // clique para selecionar trade (linhas horizontais)
    document.addEventListener('contextmenu', (e) => e.preventDefault());
    chart.subscribeClick(param => {
      if (!param || param.time === undefined) return;
      if (param?.mouseEvent?.button === 2) { clearSelection(); return; }
      const t = Number(param.time);
      const candidates = STATE.markersIndex.get(t) || [];
      if (!candidates.length) { clearSelection(); return; }
      // escolhe o mais próximo em Y
      let best = null, dyBest = Infinity;
      for (const m of candidates) {
        if (!m.__seg) continue;
        const priceAtMouse = param?.seriesPrices?.get?.(series);
        const ref = m.__seg.entry_price || m.price || 0;
        const dy = priceAtMouse == null ? 0 : Math.abs(ref - priceAtMouse);
        if (dy < dyBest) { dyBest = dy; best = m; }
      }
      if (best && best.__seg) selectTrade(best.__seg);
    });

    // mantém markers visíveis mesmo com zoom: basta não re-escalar tempos
    // já que estamos usando segundos corretos, não é preciso fazer nada aqui.
  }

  function clearSelection() {
    STATE.selectedTradeKey = null;
    for (const pl of STATE.priceLines) { try { series.removePriceLine(pl); } catch(e){} }
    STATE.priceLines = [];
  }

  function selectTrade(seg) {
    clearSelection();
    const key = tradeKey(seg);
    STATE.selectedTradeKey = key;
    const col = getTradeColor(key);

    const addLine = (price, title, color, style = LightweightCharts.LineStyle.Solid) => {
      const l = series.createPriceLine({ price, color, lineWidth: 2, lineStyle: style, axisLabelVisible: true, title });
      STATE.priceLines.push(l);
    };

    addLine(Number(seg.entry_price), 'ENTRY', col, LightweightCharts.LineStyle.Dotted);

    // usa dados da API se disponíveis; caso contrário, calcula sugestões por ATR
    let sl = seg.sl_price, tp1 = seg.tp1_price, tp2 = seg.tp2_price;
    if (!sl || !tp1 || !tp2) {
      const at = atrAt(toSec(seg.entry_time ?? seg.entry_time_ms ?? seg.entry_ts), 14);
      const f = ATR_FACT[STATE.tf] || ATR_FACT['5m'];
      if (!sl)  sl  = suggestSL(seg, at, f);
      if (!tp1) tp1 = suggestTP(seg, at, f, 1);
      if (!tp2) tp2 = suggestTP(seg, at, f, 2);
    }

    if (sl)  addLine(Number(sl),  'SL',  'rgba(239,83,80,0.9)');
    if (tp1) addLine(Number(tp1), 'TP1', 'rgba(16,185,129,0.9)', LightweightCharts.LineStyle.Dashed);
    if (tp2) addLine(Number(tp2), 'TP2', 'rgba(16,185,129,0.9)');
  }

  function tradeKey(seg) {
    // estáveis entre entradas/saídas
    return String(seg.trade_no ?? seg.id ?? `${seg.entry_time}|${seg.method ?? ''}|${seg.tf ?? ''}`);
  }

  function getTradeColor(key) {
    if (STATE.tradeColors.has(key)) return STATE.tradeColors.get(key);
    const col = COLOR_SET[STATE.tradeColors.size % COLOR_SET.length];
    STATE.tradeColors.set(key, col);
    return col;
  }

  // ------------------------- ATR no cliente (para SL/TP sugeridos quando a API não trouxe) -------------------------
  function buildAtr(data, period=14) {
    // data: candles com time, open, high, low, close
    if (!data?.length) return [];
    const atr = [];
    let prevClose = data[0].close;
    const trs = [];
    for (let i=0;i<data.length;i++){
      const c = data[i];
      const tr = Math.max(c.high - c.low, Math.abs(c.high - prevClose), Math.abs(c.low - prevClose));
      trs.push(tr);
      prevClose = c.close;
      if (trs.length >= period) {
        const slice = trs.slice(trs.length-period);
        const a = slice.reduce((s,x)=>s+x,0)/period;
        atr.push({ time: c.time, value: a });
      } else {
        atr.push({ time: c.time, value: NaN });
      }
    }
    return atr;
  }
  let ATR_CACHE = []; // {time, value}
  function atrAt(timeSec, period=14) {
    if (!timeSec) return undefined;
    if (!ATR_CACHE?.length) return undefined;
    // encontra candle <= timeSec
    let lo = 0, hi = ATR_CACHE.length-1, ans = 0;
    while (lo <= hi) {
      const mi = (lo+hi)>>1;
      if (ATR_CACHE[mi].time <= timeSec) { ans = mi; lo = mi+1; } else { hi = mi-1; }
    }
    return ATR_CACHE[ans]?.value;
  }
  function suggestSL(seg, atr, f) {
    if (!atr || !isFinite(atr)) return undefined;
    const side = (seg.side || seg.direction || 'LONG').toUpperCase();
    if (side === 'LONG') return Number(seg.entry_price) - f.sl * atr;
    return Number(seg.entry_price) + f.sl * atr;
  }
  function suggestTP(seg, atr, f, which) {
    if (!atr || !isFinite(atr)) return undefined;
    const k = which === 2 ? f.tp2 : f.tp1;
    const side = (seg.side || seg.direction || 'LONG').toUpperCase();
    if (side === 'LONG') return Number(seg.entry_price) + k * atr;
    return Number(seg.entry_price) - k * atr;
  }

  // ------------------------- UI Bindings -------------------------
  const els = {
    symbol: $('#symbol'),
    tf: $('#tf'),
    method: $('#method'),
    dir: $('#dir'),
    currency: $('#currency'),
    apply: $('#apply'),
    usdbrl: $('#usdbrl'),
    methodsInfo: $('#methodsInfo') // opcional
  };

  function syncUIFromState() {
    if (els.symbol)   els.symbol.value   = STATE.symbol;
    if (els.tf)       els.tf.value       = STATE.tf;
    if (els.method)   els.method.value   = STATE.methodId;
    if (els.dir)      els.dir.value      = STATE.dir;
    if (els.currency) els.currency.value = STATE.currency;
  }

  function persistState() {
    localStorage.setItem('tv_symbol', STATE.symbol);
    localStorage.setItem('tv_tf', STATE.tf);
    localStorage.setItem('tv_method', STATE.methodId);
    localStorage.setItem('tv_dir', STATE.dir);
    localStorage.setItem('tv_currency', STATE.currency);
  }

  function bindUI() {
    els.apply?.addEventListener('click', async () => {
      try {
        STATE.symbol   = els.symbol.value.trim().toUpperCase();
        STATE.tf       = els.tf.value;
        STATE.methodId = els.method.value;
        STATE.dir      = els.dir.value;
        STATE.currency = els.currency.value;
        persistState();

        await loadCandles();
        await loadSignals(true); // força rebuild/sort
        toast('OK', 'Configuração aplicada.');
      } catch (e) {
        alert('Falha ao aplicar: ' + e.message);
      }
    });
  }

  // ------------------------- FX -------------------------
  async function refreshFX() {
    try {
      const fx = await jget(`${API.fx}?pairs=USDTBRL`);
      // backend retorna {USDTBRL: 5.401} ou {pairs:[{pair:'USDTBRL',price:...}]}
      if (fx?.USDTBRL) STATE.fx.USDTBRL = Number(fx.USDTBRL);
      if (Array.isArray(fx?.pairs)) {
        const it = fx.pairs.find(p => p.pair === 'USDTBRL');
        if (it) STATE.fx.USDTBRL = Number(it.price);
      }
      if (els.usdbrl) els.usdbrl.textContent = (STATE.fx.USDTBRL || 0).toFixed(3);
    } catch {}
  }

  // ------------------------- Métodos (dropdown) -------------------------
  async function loadMethodsAndPopulate(tf) {
    const tfq = tf || STATE.tf;
    const raw = await jget(`${API.methods}?tf=${encodeURIComponent(tfq)}&with_scores=1`);
    // Esperado: [{method, tf, label, is_combo, score, n_trades_48h, n_trades}]
    const items = (raw || []).map(r => ({
      id: r.id || `${r.method}|${r.tf}`,
      label: r.label || `${r.method} | ${r.tf}`,
      method: r.method,
      tf: r.tf || tfq,
      is_combo: !!r.is_combo,
      score: Number(r.score ?? 0),
      trades48: Number(r.n_trades_48h ?? r.trades_48h ?? 0),
      ntrades: Number(r.n_trades ?? 0)
    }));

    // ordena: trades48 desc, depois ntrades desc, depois score desc
    items.sort((a,b) => (b.trades48 - a.trades48) || (b.ntrades - a.ntrades) || (b.score - a.score));

    // popula select
    if (els.method) {
      els.method.innerHTML = '';
      for (const it of items) {
        const opt = document.createElement('option');
        opt.value = it.id;
        const scoreStr = isFinite(it.score) && it.score !== 0 ? ` • score ${fmt.num(it.score,2)}` : '';
        const t48Str   = it.trades48 ? ` (${it.trades48} trades)` : '';
        opt.textContent = `${it.label}${t48Str}${scoreStr}`;
        els.method.appendChild(opt);
      }
    }
    STATE.methods = items;

    // se ainda não setamos um método válido, use o primeiro
    if (!STATE.methodId || !items.find(x => x.id === STATE.methodId)) {
      STATE.methodId = items[0]?.id || '';
    }
    if (els.method) els.method.value = STATE.methodId;
  }

  // ------------------------- Candles -------------------------
  async function loadCandles() {
    const limit = 2000; // suficiente para ATR + 48h
    const url = `${API.candles}?symbol=${encodeURIComponent(STATE.symbol)}&tf=${encodeURIComponent(STATE.tf)}&limit=${limit}`;
    const rows = await jget(url);
    // rows: [{time: ms, open, high, low, close, volume}]
    const data = (rows || []).map(r => ({
      time: toSec(r.time ?? r.ts ?? r.t ?? nowMs()),
      open: Number(r.open), high: Number(r.high), low: Number(r.low),
      close: Number(r.close), volume: Number(r.volume)
    })).filter(r => isFinite(r.time) && isFinite(r.open) && isFinite(r.close));

    STATE.candles = data;
    series.setData(data);
    if (volSeries) {
      const vd = data.map(c => ({ time: c.time, value: c.volume }));
      volSeries.setData(vd);
    }

    // posiciona range nos últimos ~72h
    const end = data[data.length - 1]?.time;
    const from = end ? (end - 72 * 3600) : undefined;
    if (end && from) chart.timeScale().setVisibleRange({ from, to: end });

    // recalcula ATR para SL/TP sugeridos
    ATR_CACHE = buildAtr(data, 14);
  }

  // ------------------------- Signals (48h + live) -------------------------
  async function loadSignals(sortMethods = false) {
    // janela 48h
    const to   = nowMs();
    const from = to - 48 * 3600 * 1000;
    const params = new URLSearchParams({
      symbol: STATE.symbol,
      method_id: STATE.methodId,
      method: (STATE.methodId.includes('|') ? STATE.methodId.split('|')[0] : STATE.methodId),
      tf: STATE.tf,
      from: String(from),
      to: String(to),
      full: '1',
      dir: STATE.dir
    });
    const url = `${API.signals}?${params.toString()}`;
    const payload = await jget(url);
    // payload esperado: {entries:[], exits:[], segments:[]}
    STATE.signals = payload || {};
    STATE.segments = Array.isArray(payload?.segments) ? payload.segments : [];

    // monta markers
    const markers = [];
    const idx = new Map();
    let plUsd48 = 0;

    for (const seg of STATE.segments) {
      const entryT = toSec(seg.entry_time_ms ?? seg.entry_time ?? seg.entry_ts);
      if (!isFinite(entryT)) continue;

      const side = (seg.side || seg.direction || 'LONG').toUpperCase();
      const key = tradeKey(seg);
      const col = getTradeColor(key);

      // entrada
      markers.push({
        time: entryT,
        position: side === 'LONG' ? 'belowBar' : 'aboveBar',
        color: col,
        shape: side === 'LONG' ? 'arrowUp' : 'arrowDown',
        text: `${side} #${seg.trade_no ?? ''}`.trim(),
        __seg: seg
      });
      pushIdx(idx, entryT, markers[markers.length - 1]);

      // saída (se houver)
      const exT = toSec(seg.exit_time_ms ?? seg.exit_time ?? seg.exit_ts);
      if (isFinite(exT)) {
        const reason = (seg.exit_reason || '').toUpperCase();
        const shape = reason.startsWith('TP') ? 'circle' : (reason.startsWith('SL') ? 'square' : 'circle');
        markers.push({
          time: exT,
          position: side === 'LONG' ? 'aboveBar' : 'belowBar',
          color: col,
          shape,
          text: `${reason || 'EXIT'} #${seg.trade_no ?? ''}`.trim(),
          __seg: seg
        });
        pushIdx(idx, exT, markers[markers.length - 1]);
      }

      // P/L 48h
      plUsd48 += Number(seg.pnl_usd ?? seg.pnl ?? 0);
    }

    STATE.markers = markers;
    series.setMarkers(markers);
    STATE.markersIndex = idx;

    // badge 48h
    const rate = STATE.currency === 'BRL' ? (STATE.fx.USDTBRL || 1) : 1;
    const pl = plUsd48 * rate;
    const wins = STATE.segments.filter(s => Number(s.pnl_usd ?? s.pnl ?? 0) > 0).length;
    const tot  = STATE.segments.length;
    if (plBadgeEl) {
      const money = STATE.currency === 'BRL' ? fmt.brl(pl) : fmt.usd(plUsd48);
      plBadgeEl.textContent = `48h P/L: ${fmt.pct(0)} • ${money} • ${wins}/${tot} W`; // pct 0 por enquanto (sem eq)
    }

    // opcional: reordenar métodos usando os trades desta consulta
    if (sortMethods) {
      const counts = new Map(STATE.methods.map(m => [m.id, 0]));
      counts.set(STATE.methodId, STATE.segments.length);
      // Atualiza label do método corrente com (#trades)
      if (els.method) {
        const opt = Array.from(els.method.options).find(o => o.value === STATE.methodId);
        if (opt) {
          const base = opt.textContent.replace(/\(\d+ trades\).*/,'').trim();
          opt.textContent = `${base} (${STATE.segments.length} trades)`;
        }
      }
    }
  }

  function pushIdx(map, t, m) {
    if (!map.has(t)) map.set(t, []);
    map.get(t).push(m);
  }

  // ------------------------- Polling Live -------------------------
  function startPolling() {
    setInterval(refreshFX, 30_000);
    setInterval(async () => {
      try {
        await loadCandles();
        await loadSignals();
      } catch {}
    }, 20_000); // 20s: sinais/candles atualizados
  }

  // ------------------------- Boot -------------------------
  async function boot() {
    try {
      syncUIFromState();
      bindUI();
      initCharts();

      await refreshFX();
      await loadMethodsAndPopulate(STATE.tf);
      await loadCandles();
      await loadSignals(true);

      STATE.bootDone = true;
      console.debug('[boot] ok', {
        symbol: STATE.symbol, tf: STATE.tf, method: STATE.methodId,
        currency: STATE.currency, dir: STATE.dir, markers: STATE.markers.length
      });
      startPolling();
    } catch (e) {
      console.error('[boot] error:', e);
      alert('Erro no boot: ' + e.message);
    }
  }

  // ------------------------- Toast (mínimo) -------------------------
  function toast(title, msg) {
    console.log(`[${title}] ${msg}`);
  }

  // ------------------------- Resize -------------------------
  function onResize() {
    if (!chart) return;
    const w = chartEl.clientWidth, h = chartEl.clientHeight;
    chart.applyOptions({ width: w, height: h });
    if (volChart) volChart.applyOptions({ width: w });
  }
  window.addEventListener('resize', onResize);

  // ------------------------- Inicia -------------------------
  document.addEventListener('DOMContentLoaded', boot);
})();
