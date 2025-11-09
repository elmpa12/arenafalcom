const API_BASE = (typeof import.meta !== 'undefined' && import.meta.env && import.meta.env.VITE_API_BASE)
  ? import.meta.env.VITE_API_BASE.replace(/\/$/, '')
  : '';
const PAGE_SIZE = 2000;
const SPEED_MAP = {
  1: 550,
  2: 330,
  5: 180,
  10: 90,
};

const PIPELINE_ALL = '__all__';
const PIPELINE_UNTAGGED = '__sem_pipeline__';

const PIPELINE_TYPE_LABEL = {
  base: 'Base',
  ml: 'ML',
  dl: 'DL',
  ensemble: 'Ensemble',
  custom: 'Personalizado',
};

const PIPELINE_TYPE_CLASS = {
  base: 'type-base',
  ml: 'type-ml',
  dl: 'type-dl',
  ensemble: 'type-ensemble',
  custom: 'type-custom',
};

const fmt = {
  pct: (value) => (Number.isFinite(value) ? (value * 100).toFixed(1) + '%' : '—'),
  num: (value) => (Number.isFinite(value) ? value.toFixed(2) : '—'),
  int: (value) => (Number.isFinite(value) ? Math.round(value).toString() : '—'),
  edge: (value) => (Number.isFinite(value) ? `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%` : '—'),
};

function slugify(value) {
  if (!value) return 'pipeline';
  return value
    .toString()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .trim() || 'pipeline';
}

function normalisePipelineDef(raw) {
  if (!raw || typeof raw !== 'object') {
    return null;
  }
  const name = raw.name || raw.nome || raw.display_name || raw.id || 'Pipeline';
  const id = raw.id || slugify(name);
  const type = (raw.type || raw.tipo || 'custom').toString().toLowerCase();
  return {
    id,
    name,
    type,
    description: raw.description || raw.descricao || '',
    edge: Number.isFinite(raw.edge) ? raw.edge : Number.isFinite(raw.alpha) ? raw.alpha : null,
    kpis: raw.kpis || raw.metrics || {},
  };
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json();
}

function toUnix(ts) {
  return Math.floor(new Date(ts).getTime() / 1000);
}

class KPITracker {
  constructor() {
    this.trades = [];
    this.equity = [];
    this.syntheticEquity = [0];
  }

  update(frame, useEquity = true) {
    if (useEquity && typeof frame.equity === 'number') {
      this.equity.push(frame.equity);
    }
    const closed = frame.trades_closed || [];
    if (closed.length) {
      let base = this.syntheticEquity[this.syntheticEquity.length - 1] ?? 0;
      for (const trade of closed) {
        this.trades.push(trade);
        const pnl = Number.isFinite(trade.pnl) ? trade.pnl : 0;
        base += pnl;
        this.syntheticEquity.push(base);
      }
    }
  }

  snapshot() {
    return {
      winrate: this._winRate(),
      sharpe: this._sharpe(),
      max_drawdown: this._maxDrawdown(),
      profit_factor: this._profitFactor(),
      expectancy: this._expectancy(),
      avg_trade: this._expectancy(),
      n_trades: this.trades.length,
      hit_long: this._hitRatio('LONG'),
      hit_short: this._hitRatio('SHORT'),
    };
  }

  _closedTrades() {
    return this.trades.filter((t) => typeof t.pnl === 'number');
  }

  _winRate() {
    const closed = this._closedTrades();
    if (!closed.length) return 0;
    const wins = closed.filter((t) => t.pnl > 0).length;
    return wins / closed.length;
  }

  _profitFactor() {
    const closed = this._closedTrades();
    let gains = 0;
    let losses = 0;
    for (const t of closed) {
      if (t.pnl > 0) gains += t.pnl;
      if (t.pnl < 0) losses += Math.abs(t.pnl);
    }
    if (losses === 0) return gains > 0 ? Infinity : 0;
    return gains / losses;
  }

  _maxDrawdown() {
    const series =
      this.equity.length >= 2
        ? this.equity
        : this.syntheticEquity.length >= 2
        ? this.syntheticEquity
        : [];
    if (series.length < 2) return 0;
    let peak = series[0];
    let maxDd = 0;
    for (const e of series) {
      if (e > peak) peak = e;
      const dd = peak - e;
      if (dd > maxDd) maxDd = dd;
    }
    return maxDd;
  }

  _expectancy() {
    const closed = this._closedTrades();
    if (!closed.length) return 0;
    const sum = closed.reduce((acc, t) => acc + (t.pnl ?? 0), 0);
    return sum / closed.length;
  }

  _hitRatio(side) {
    const closed = this._closedTrades().filter((t) => t.side === side);
    if (!closed.length) return 0;
    const wins = closed.filter((t) => t.pnl > 0).length;
    return wins / closed.length;
  }

  _sharpe() {
    const series =
      this.equity.length >= 2
        ? this.equity
        : this.syntheticEquity.length >= 2
        ? this.syntheticEquity
        : [];
    if (series.length < 2) return 0;
    const returns = [];
    for (let i = 1; i < series.length; i += 1) {
      const prev = series[i - 1];
      const curr = series[i];
      const base = Math.abs(prev) > 1e-9 ? Math.abs(prev) : 1.0;
      returns.push((curr - prev) / base);
    }
    if (!returns.length) return 0;
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length;
    if (variance <= 0) return 0;
    return (mean / Math.sqrt(variance)) * Math.sqrt(252);
  }
}

class ReplayPane {
  constructor(side) {
    this.side = side;
    const suffix = side.toLowerCase();
    this.container = document.getElementById(`chart-${suffix}`);
    this.equityContainer = document.getElementById(`equity-${suffix}`);
    this.metaEl = document.getElementById(`meta-${suffix}`);
    this.pipelineContainer = document.getElementById(`pipelines-${suffix}`);
    this.kpiEl = document.getElementById(`kpis-${suffix}`);
    this.tradesEl = document.getElementById(`trades-${suffix}`);
    this.chart = null;
    this.candleSeries = null;
    this.markerBuffer = [];
    this.equityChart = null;
    this.equitySeries = null;
    this.frames = [];
    this.total = 0;
    this.playhead = 0;
    this.meta = null;
    this.tracker = new KPITracker();
    this.trades = [];
    this.closedTradesBuffer = [];
    this.pipelineIndex = new Map();
    this.pipelineOrder = [];
    this.pipelineFilter = new Set([PIPELINE_ALL]);
    this.hasUntagged = false;
    this.initialised = false;
    this._resizeHandler = null;
  }

  async select(backtestId) {
    this.reset();
    if (!backtestId) return;
    this.backtestId = backtestId;
    this.meta = await fetchJSON(`${API_BASE}/api/backtests/${backtestId}/meta`);
    this.total = this.meta.n_frames;
    const tradePayload = await fetchJSON(`${API_BASE}/api/backtests/${backtestId}/trades`);
    this.trades = tradePayload.trades || [];
    this.preparePipelines();
    this.updateMeta();
    await this.ensureIndex(0);
    this.bootstrapCharts();
    this.redraw();
  }

  reset() {
    this.frames = [];
    this.tracker = new KPITracker();
    this.playhead = 0;
    this.metaEl.textContent = '';
    this.closedTradesBuffer = [];
    this.pipelineIndex = new Map();
    this.pipelineOrder = [];
    this.pipelineFilter = new Set([PIPELINE_ALL]);
    this.hasUntagged = false;
    if (this.pipelineContainer) {
      this.pipelineContainer.innerHTML = '';
    }
    this.tradesEl.innerHTML = '';
    if (this.chart) {
      if (this._resizeHandler) {
        window.removeEventListener('resize', this._resizeHandler);
        this._resizeHandler = null;
      }
      this.chart.remove();
    }
    if (this.equityChart) {
      this.equityChart.remove();
    }
    this.chart = null;
    this.equityChart = null;
    this.candleSeries = null;
    this.equitySeries = null;
    this.markerBuffer = [];
    this.initialised = false;
  }

  bootstrapCharts() {
    const options = {
      layout: {
        background: { color: '#10141c' },
        textColor: '#ccd6f6',
      },
      grid: {
        vertLines: { color: 'rgba(197, 214, 255, 0.12)' },
        horzLines: { color: 'rgba(197, 214, 255, 0.12)' },
      },
      timeScale: { rightOffset: 2, barSpacing: 8, fixLeftEdge: true },
      crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    };
    this.chart = LightweightCharts.createChart(this.container, options);
    this.candleSeries = this.chart.addCandlestickSeries({
      upColor: '#4ee7a4',
      downColor: '#ff6b6b',
      borderUpColor: '#4ee7a4',
      borderDownColor: '#ff6b6b',
      wickUpColor: '#4ee7a4',
      wickDownColor: '#ff6b6b',
    });
    this.equityChart = LightweightCharts.createChart(this.equityContainer, {
      ...options,
      crosshair: { mode: LightweightCharts.CrosshairMode.Hidden },
      timeScale: { rightOffset: 2, barSpacing: 8 },
    });
    this.equitySeries = this.equityChart.addLineSeries({ color: '#4cc9f0', lineWidth: 2 });
    const resize = () => {
      this.chart.applyOptions({ width: this.container.clientWidth, height: this.container.clientHeight });
      this.equityChart.applyOptions({ width: this.equityContainer.clientWidth, height: this.equityContainer.clientHeight });
    };
    resize();
    this._resizeHandler = resize;
    window.addEventListener('resize', this._resizeHandler);
    this.initialised = true;
  }

  async ensureIndex(index) {
    if (index < this.frames.length && this.frames[index]) return;
    while (this.frames.length <= index && this.frames.length < this.total) {
      const offset = this.frames.length;
      const payload = await fetchJSON(`${API_BASE}/api/backtests/${this.backtestId}/frames?offset=${offset}&limit=${PAGE_SIZE}`);
      const rows = payload.frames || [];
      if (!rows.length) break;
      this.frames.push(...rows);
    }
  }

  async step() {
    if (this.playhead >= this.total) return;
    await this.ensureIndex(this.playhead);
    const frame = this.frames[this.playhead];
    if (!frame) return;
    this.applyFrame(frame);
    this.playhead += 1;
    updateGlobalTime();
  }

  async seek(index) {
    if (!this.total) return;
    const clamped = Math.max(0, Math.min(index, this.total - 1));
    await this.ensureIndex(clamped);
    this.playhead = clamped;
    this.redraw();
    updateGlobalTime();
  }

  applyFrame(frame) {
    if (!this.initialised) return;
    const bar = frame.bar;
    const time = toUnix(bar.t);
    const candle = {
      time,
      open: bar.o,
      high: bar.h,
      low: bar.l,
      close: bar.c,
    };
    this.candleSeries.update(candle);
    if (typeof frame.equity === 'number') {
      this.equitySeries.update({ time, value: frame.equity });
    }

    const allowedSignals = [];
    if (frame.signals && frame.signals.length) {
      for (const signal of frame.signals) {
        const key = this.registerPipelineIfUnknown(signal.pipeline);
        if (!this.pipelineAllowedNormalized(key)) continue;
        allowedSignals.push({ ...signal, pipeline: key });
        this.markerBuffer.push({
          time,
          position: signal.side === 'BUY' ? 'belowBar' : 'aboveBar',
          color: signal.side === 'BUY' ? '#4ee7a4' : '#ff6b6b',
          shape: signal.side === 'BUY' ? 'arrowUp' : 'arrowDown',
          text: signal.reason || signal.side,
        });
      }
      this.candleSeries.setMarkers(this.markerBuffer.slice(-200));
    }

    const allowedClosed = [];
    if (frame.trades_closed && frame.trades_closed.length) {
      for (const trade of frame.trades_closed) {
        const key = this.registerPipelineIfUnknown(trade.pipeline || trade.meta?.pipeline, {
          name: trade.meta?.pipeline_name,
          type: trade.meta?.pipeline_type,
          description: trade.meta?.pipeline_description,
          edge: trade.meta?.edge,
          kpis: trade.meta?.kpis,
        });
        const decorated = { ...trade, pipeline: key };
        if (this.pipelineAllowedNormalized(key)) {
          allowedClosed.push(decorated);
          this.closedTradesBuffer.push(decorated);
        }
      }
      this.renderTrades(this.closedTradesBuffer);
    }

    const filteredFrame = {
      ...frame,
      signals: allowedSignals,
      trades_closed: allowedClosed,
    };
    const useEquity = this.pipelineFilter.has(PIPELINE_ALL);
    this.tracker.update(filteredFrame, useEquity);
    this.renderKpis();
  }

  redraw() {
    if (!this.initialised) return;
    const candles = [];
    const equities = [];
    this.markerBuffer = [];
    this.tracker = new KPITracker();
    this.closedTradesBuffer = [];
    const useEquity = this.pipelineFilter.has(PIPELINE_ALL);
    for (let i = 0; i <= this.playhead && i < this.frames.length; i += 1) {
      const frame = this.frames[i];
      const bar = frame.bar;
      const time = toUnix(bar.t);
      candles.push({ time, open: bar.o, high: bar.h, low: bar.l, close: bar.c });
      if (typeof frame.equity === 'number') {
        equities.push({ time, value: frame.equity });
      }

      const allowedSignals = [];
      if (frame.signals) {
        for (const signal of frame.signals) {
          const key = this.registerPipelineIfUnknown(signal.pipeline);
          if (!this.pipelineAllowedNormalized(key)) continue;
          allowedSignals.push({ ...signal, pipeline: key });
          this.markerBuffer.push({
            time,
            position: signal.side === 'BUY' ? 'belowBar' : 'aboveBar',
            color: signal.side === 'BUY' ? '#4ee7a4' : '#ff6b6b',
            shape: signal.side === 'BUY' ? 'arrowUp' : 'arrowDown',
            text: signal.reason || signal.side,
          });
        }
      }

      const allowedClosed = [];
      if (frame.trades_closed) {
        for (const trade of frame.trades_closed) {
          const key = this.registerPipelineIfUnknown(trade.pipeline || trade.meta?.pipeline, {
            name: trade.meta?.pipeline_name,
            type: trade.meta?.pipeline_type,
            description: trade.meta?.pipeline_description,
            edge: trade.meta?.edge,
            kpis: trade.meta?.kpis,
          });
          const decorated = { ...trade, pipeline: key };
          if (this.pipelineAllowedNormalized(key)) {
            allowedClosed.push(decorated);
            this.closedTradesBuffer.push(decorated);
          }
        }
      }

      const filteredFrame = {
        ...frame,
        signals: allowedSignals,
        trades_closed: allowedClosed,
      };
      this.tracker.update(filteredFrame, useEquity);
    }
    this.candleSeries.setData(candles);
    this.equitySeries.setData(equities);
    this.candleSeries.setMarkers(this.markerBuffer.slice(-200));
    this.renderTrades(this.closedTradesBuffer);
    this.renderKpis();
  }

  renderKpis() {
    const snap = this.tracker.snapshot();
    const items = [
      { key: 'Taxa de Acerto', value: fmt.pct(snap.winrate) },
      { key: 'Sharpe', value: fmt.num(snap.sharpe) },
      { key: 'Máx. Drawdown', value: fmt.num(snap.max_drawdown) },
      { key: 'Profit Factor', value: fmt.num(snap.profit_factor) },
      { key: 'Expectativa', value: fmt.num(snap.expectancy) },
      { key: 'Média por Trade', value: fmt.num(snap.avg_trade) },
      { key: 'Nº de Trades', value: fmt.int(snap.n_trades) },
      { key: 'Taxa LONG', value: fmt.pct(snap.hit_long) },
      { key: 'Taxa SHORT', value: fmt.pct(snap.hit_short) },
    ];
    this.kpiEl.innerHTML = items
      .map(
        (item) => `
          <div class="card">
            <span class="label">${item.key}</span>
            <span class="value">${item.value}</span>
          </div>
        `,
      )
      .join('');
  }

  renderTrades(trades) {
    if (!this.tradesEl) return;
    const recent = trades.slice(-25).reverse();
    this.tradesEl.innerHTML = `
      <table>
        <thead>
          <tr>
            <th>Entrada</th>
            <th>Saída</th>
            <th>Direção</th>
            <th>Preço Entrada</th>
            <th>Preço Saída</th>
            <th>PnL</th>
            <th>Pipeline</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    `;
    const tbody = this.tradesEl.querySelector('tbody');
    if (!recent.length) {
      const row = document.createElement('tr');
      row.innerHTML = '<td colspan="7">Nenhum trade filtrado até agora.</td>';
      tbody.appendChild(row);
      return;
    }
    for (const trade of recent) {
      const row = document.createElement('tr');
      const pnlClass = trade.pnl > 0 ? 'badge-win' : trade.pnl < 0 ? 'badge-loss' : '';
      row.innerHTML = `
        <td>${trade.entry_t}</td>
        <td>${trade.exit_t ?? '—'}</td>
        <td>${trade.side}</td>
        <td>${fmt.num(trade.entry_px)}</td>
        <td>${fmt.num(trade.exit_px ?? Number.NaN)}</td>
        <td class="${pnlClass}">${fmt.num(trade.pnl ?? Number.NaN)}</td>
        <td>${this.pipelineLabel(trade.pipeline)}</td>
      `;
      tbody.appendChild(row);
    }
  }

  updateMeta() {
    if (!this.meta || !this.metaEl) return;
    const unique = new Set(this.pipelineOrder);
    if (this.hasUntagged) {
      unique.add(PIPELINE_UNTAGGED);
    }
    const count = unique.size || 0;
    const label = count === 1 ? 'pipeline' : 'pipelines';
    this.metaEl.innerHTML = `
      <strong>${this.meta.symbol}</strong> • ${this.meta.timeframe} • ${this.meta.start} → ${this.meta.end} • ${this.meta.n_frames} barras • ${this.meta.n_trades} trades • ${count} ${label}
    `;
  }

  async findTradeIndex(direction) {
    const increment = direction > 0 ? 1 : -1;
    let index = this.playhead + increment;
    while (index >= 0 && index < this.total) {
      await this.ensureIndex(index);
      const frame = this.frames[index];
      if (frame && frame.trades_closed && frame.trades_closed.length) {
        const match = frame.trades_closed.some((trade) => {
          const key = this.registerPipelineIfUnknown(trade.pipeline || trade.meta?.pipeline, {
            name: trade.meta?.pipeline_name,
            type: trade.meta?.pipeline_type,
            description: trade.meta?.pipeline_description,
            edge: trade.meta?.edge,
            kpis: trade.meta?.kpis,
          });
          return this.pipelineAllowedNormalized(key);
        });
        if (match) {
          return index;
        }
      }
      index += increment;
    }
    return null;
  }

  preparePipelines() {
    this.pipelineIndex = new Map();
    this.pipelineOrder = [];
    this.pipelineFilter = new Set([PIPELINE_ALL]);
    this.hasUntagged = false;
    const baseList = Array.isArray(this.meta?.pipelines)
      ? this.meta.pipelines
      : Array.isArray(this.meta?.params?.pipelines)
      ? this.meta.params.pipelines
      : [];
    for (const raw of baseList) {
      const def = normalisePipelineDef(raw);
      if (def) {
        this.addPipelineDefinition(def);
      }
    }
    for (const trade of this.trades) {
      const key = this.normalizePipelineKey(trade.pipeline || trade.meta?.pipeline);
      if (key !== PIPELINE_UNTAGGED) {
        this.addPipelineDefinition({
          id: key,
          name: trade.meta?.pipeline_name || trade.pipeline || key,
          type: (trade.meta?.pipeline_type || 'custom').toString().toLowerCase(),
          description: trade.meta?.pipeline_description || '',
          edge: Number.isFinite(trade.meta?.edge) ? trade.meta.edge : null,
          kpis: trade.meta?.kpis || {},
        });
      }
    }
    this.renderPipelineCards();
  }

  normalizePipelineKey(id, { markUntagged = true } = {}) {
    if (!id) {
      if (markUntagged) this.hasUntagged = true;
      return PIPELINE_UNTAGGED;
    }
    const key = id.toString();
    if (key === PIPELINE_ALL) {
      return `${PIPELINE_ALL}-origem`;
    }
    return key;
  }

  registerPipelineIfUnknown(rawId, options = {}) {
    const prevUntagged = this.hasUntagged;
    const key = this.normalizePipelineKey(rawId);
    if (key === PIPELINE_UNTAGGED) {
      if (!prevUntagged && this.hasUntagged) {
        this.renderPipelineCards();
      }
      return key;
    }
    const payload = {
      id: key,
      name: options.name || (typeof rawId === 'string' ? rawId : key),
      type: (options.type || 'custom').toString().toLowerCase(),
      description: options.description || '',
      edge: Number.isFinite(options.edge) ? options.edge : null,
      kpis: options.kpis || {},
    };
    const exists = this.pipelineIndex.has(key);
    this.addPipelineDefinition(payload);
    if (!exists) {
      this.renderPipelineCards();
    }
    return key;
  }

  addPipelineDefinition(def) {
    if (!def || !def.id) return;
    const normalized = {
      id: def.id,
      name: def.name || def.id,
      type: (def.type || 'custom').toString().toLowerCase(),
      description: def.description || '',
      edge: Number.isFinite(def.edge) ? def.edge : null,
      kpis: def.kpis || {},
    };
    if (this.pipelineIndex.has(normalized.id)) {
      const previous = this.pipelineIndex.get(normalized.id);
      this.pipelineIndex.set(normalized.id, {
        ...previous,
        ...normalized,
        kpis: { ...(previous?.kpis || {}), ...(normalized.kpis || {}) },
      });
    } else {
      this.pipelineIndex.set(normalized.id, normalized);
      this.pipelineOrder.push(normalized.id);
    }
  }

  ensureUntaggedDefinition() {
    if (!this.hasUntagged || this.pipelineIndex.has(PIPELINE_UNTAGGED)) {
      return;
    }
    this.addPipelineDefinition({
      id: PIPELINE_UNTAGGED,
      name: 'Sem classificação',
      type: 'custom',
      description: 'Eventos sem pipeline identificado.',
      edge: null,
      kpis: {},
    });
  }

  renderPipelineCards() {
    if (!this.pipelineContainer) return;
    this.ensureUntaggedDefinition();
    const seen = new Set();
    const cards = [];
    cards.push(
      this.pipelineCardTemplate(
        {
          id: PIPELINE_ALL,
          name: 'Todos os pipelines',
          type: 'custom',
          description: 'Exibir sinais de todas as origens.',
          edge: null,
          kpis: {},
        },
        true,
      ),
    );
    for (const id of this.pipelineOrder) {
      if (seen.has(id)) continue;
      const def = this.pipelineIndex.get(id);
      if (!def) continue;
      seen.add(id);
      cards.push(this.pipelineCardTemplate(def));
    }
    this.pipelineContainer.innerHTML = cards.join('');
    this.pipelineContainer.querySelectorAll('[data-action="pipeline-toggle"]').forEach((el) => {
      el.addEventListener('click', () => {
        this.togglePipeline(el.dataset.pipelineId);
      });
    });
    this.updatePipelineCardStates();
    this.updateMeta();
  }

  pipelineCardTemplate(def, isAll = false) {
    const typeKey = (def.type || 'custom').toLowerCase();
    const cssClass = PIPELINE_TYPE_CLASS[typeKey] || PIPELINE_TYPE_CLASS.custom;
    const tag = isAll ? 'Todos' : PIPELINE_TYPE_LABEL[typeKey] || typeKey.toUpperCase();
    const edge = !isAll && Number.isFinite(def.edge) ? `<span class="edge">Edge ${fmt.edge(def.edge)}</span>` : '';
    const metrics = [];
    if (def.kpis && Number.isFinite(def.kpis.winrate)) {
      metrics.push(`<span>Taxa de acerto ${fmt.pct(def.kpis.winrate)}</span>`);
    }
    if (def.kpis && Number.isFinite(def.kpis.expectancy)) {
      metrics.push(`<span>Expectativa ${fmt.num(def.kpis.expectancy)}</span>`);
    }
    const metricsBlock = metrics.length ? `<div class="mini-metrics">${metrics.join('')}</div>` : '';
    const description = def.description ? `<p class="desc">${def.description}</p>` : '';
    return `
      <button type="button" class="pipeline-card ${cssClass}" data-action="pipeline-toggle" data-pipeline-id="${def.id}" data-active="false">
        <span class="tag">${tag}</span>
        <span class="title">${def.name}</span>
        ${edge}
        ${metricsBlock}
        ${description}
      </button>
    `;
  }

  togglePipeline(id) {
    if (!id) return;
    if (id === PIPELINE_ALL) {
      this.pipelineFilter = new Set([PIPELINE_ALL]);
    } else {
      const next = new Set(this.pipelineFilter);
      if (next.has(PIPELINE_ALL)) {
        next.delete(PIPELINE_ALL);
      }
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      if (!next.size) {
        next.add(PIPELINE_ALL);
      }
      this.pipelineFilter = next;
    }
    this.updatePipelineCardStates();
    this.redraw();
  }

  updatePipelineCardStates() {
    if (!this.pipelineContainer) return;
    const showAll = this.pipelineFilter.has(PIPELINE_ALL);
    this.pipelineContainer.querySelectorAll('[data-action="pipeline-toggle"]').forEach((el) => {
      const id = el.dataset.pipelineId;
      const active = showAll ? true : this.pipelineFilter.has(id);
      el.setAttribute('data-active', active ? 'true' : 'false');
    });
  }

  pipelineAllowedNormalized(key) {
    if (this.pipelineFilter.has(PIPELINE_ALL)) return true;
    return this.pipelineFilter.has(key);
  }

  pipelineLabel(rawId) {
    const key = this.normalizePipelineKey(rawId, { markUntagged: false });
    if (key === PIPELINE_UNTAGGED) return 'Sem classificação';
    const def = this.pipelineIndex.get(key);
    return def?.name || (typeof rawId === 'string' ? rawId : '—');
  }
}

class PlaybackController {
  constructor(players) {
    this.players = players;
    this.speed = 1;
    this.timer = null;
    this.running = false;
  }

  setSpeed(value) {
    this.speed = Number(value) || 1;
    if (this.running) {
      this.pause();
      this.play();
    }
  }

  play() {
    if (this.running) return;
    this.running = true;
    const loop = async () => {
      if (!this.running) return;
      await Promise.all(Object.values(this.players).map((p) => p.step()));
      this.timer = setTimeout(loop, SPEED_MAP[this.speed] || 500);
    };
    loop();
  }

  pause() {
    this.running = false;
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  async seek(progress) {
    const ratio = progress / 1000;
    await Promise.all(
      Object.values(this.players).map((player) => {
        const target = Math.floor(player.total * ratio);
        return player.seek(target);
      }),
    );
  }

  async jumpTrade(direction) {
    await Promise.all(
      Object.values(this.players).map(async (player) => {
        const idx = await player.findTradeIndex(direction);
        if (idx !== null) {
          await player.seek(idx);
        }
      }),
    );
  }
}

function updateGlobalTime() {
  const label = document.getElementById('time-label');
  const player = players.A;
  if (!player || !player.total || !player.frames[player.playhead]) {
    label.textContent = '--';
    return;
  }
  label.textContent = player.frames[player.playhead].bar.t;
  const scrubber = document.getElementById('scrubber');
  const ratio = player.playhead / Math.max(1, player.total);
  scrubber.value = Math.round(ratio * 1000);
}

async function bootstrap() {
  const list = await fetchJSON(`${API_BASE}/api/backtests`);
  const selectA = document.getElementById('dataset-a');
  const selectB = document.getElementById('dataset-b');
  for (const item of list) {
    const option1 = document.createElement('option');
    option1.value = item.id;
    option1.textContent = `${item.id} (${item.symbol} ${item.timeframe})`;
    selectA.appendChild(option1);
    const option2 = document.createElement('option');
    option2.value = item.id;
    option2.textContent = `${item.id} (${item.symbol} ${item.timeframe})`;
    selectB.appendChild(option2);
  }
  if (list[0]) selectA.value = list[0].id;
  if (list[1]) selectB.value = list[1].id;
  await Promise.all([
    players.A.select(selectA.value || list[0]?.id || ''),
    players.B.select(selectB.value || list[1]?.id || ''),
  ]);
  setupEvents();
}

function setupEvents() {
  const playBtn = document.getElementById('btn-play');
  const pauseBtn = document.getElementById('btn-pause');
  const speedSelect = document.getElementById('speed');
  const scrubber = document.getElementById('scrubber');
  const nextBtn = document.getElementById('btn-next');
  const prevBtn = document.getElementById('btn-prev');
  const selectA = document.getElementById('dataset-a');
  const selectB = document.getElementById('dataset-b');

  playBtn.addEventListener('click', () => playback.play());
  pauseBtn.addEventListener('click', () => playback.pause());
  speedSelect.addEventListener('change', (event) => playback.setSpeed(event.target.value));
  scrubber.addEventListener('input', (event) => {
    playback.seek(Number(event.target.value));
  });
  nextBtn.addEventListener('click', () => {
    playback.jumpTrade(1);
  });
  prevBtn.addEventListener('click', () => {
    playback.jumpTrade(-1);
  });
  selectA.addEventListener('change', (event) => {
    players.A.select(event.target.value);
  });
  selectB.addEventListener('change', (event) => {
    players.B.select(event.target.value);
  });

  window.addEventListener('keydown', (event) => {
    if (event.code === 'Space') {
      event.preventDefault();
      if (playback.running) playback.pause();
      else playback.play();
    }
    if (event.code === 'ArrowRight') {
      event.preventDefault();
      if (event.shiftKey) playback.jumpTrade(1);
      else playback.seek(Math.min(1000, Number(scrubber.value) + 5));
    }
    if (event.code === 'ArrowLeft') {
      event.preventDefault();
      if (event.shiftKey) playback.jumpTrade(-1);
      else playback.seek(Math.max(0, Number(scrubber.value) - 5));
    }
  });
}

const players = {
  A: new ReplayPane('A'),
  B: new ReplayPane('B'),
};

const playback = new PlaybackController(players);

bootstrap().catch((err) => {
  const app = document.getElementById('app');
  app.innerHTML = `<div class="error">Falha ao carregar os backtests: ${err.message}</div>`;
});
