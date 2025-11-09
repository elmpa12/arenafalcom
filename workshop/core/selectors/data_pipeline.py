"""Shared data enrichment pipeline for selector21 and deep-learning heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import glob
import io
import os
import re
import zipfile

import numpy as np
import pandas as pd


@dataclass(slots=True)
class EnrichmentRequest:
    """Parameters required to assemble an enriched feature frame."""

    symbol: str
    timeframe: str
    start_ms: int
    end_ms: int
    root_dir: str
    args: Optional[Any] = None
    base_data: Optional[Dict[str, pd.DataFrame]] = None


def _tf_to_rule(tf_: str) -> str:
    tf_ = str(tf_).lower().strip()
    if tf_.endswith("m"):
        return f"{int(tf_[:-1])}min"
    if tf_.endswith("h"):
        return f"{int(tf_[:-1])}H"
    if tf_.endswith("d"):
        return f"{int(tf_[:-1])}D"
    return "5min"


def _to_utc(value, unit: Optional[str] = None) -> pd.Timestamp:
    try:
        return pd.to_datetime(value, utc=True, unit=unit, errors="coerce")
    except Exception:
        return pd.to_datetime(value, utc=True, errors="coerce")


def _clip_std(series: pd.Series, k: float = 8.0) -> pd.Series:
    mean = series.mean()
    std = series.std(ddof=0)
    if std == 0 or np.isnan(std):
        return series
    low, high = mean - k * std, mean + k * std
    return series.clip(lower=low, upper=high)


def _downcast_inplace(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=[np.floating]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=[np.integer]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(series: pd.Series, length: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs = up.rolling(length).mean() / (down.rolling(length).mean() + 1e-9)
    return 100 - (100 / (1 + rs))


def _boll_pos(series: pd.Series, length: int = 20, mult: float = 2) -> pd.Series:
    mean = series.rolling(length).mean()
    std = series.rolling(length).std(ddof=0)
    return (series - mean) / (mult * std + 1e-9)


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(length).mean()


def _pick_metrics_time_col(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "create_time",
        "time",
        "timestamp",
        "datetime",
        "time_ms",
        "timestamp_ms",
    ]
    cols = [str(c) for c in df.columns]
    normalised = {re.sub(r"[^a-z0-9]", "", c.lower()): c for c in cols}
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        if key in normalised:
            return normalised[key]
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        for norm, original in normalised.items():
            if norm.startswith(key) or key in norm:
                return original
    return None


def _read_zip_any_csv(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    try:
        with zipfile.ZipFile(path) as zf:
            names = [
                name
                for name in zf.namelist()
                if name.lower().endswith(".csv") and not name.endswith("/")
            ]
            target = names[0] if names else next(
                (name for name in zf.namelist() if not name.endswith("/")),
                None,
            )
            if not target:
                return pd.DataFrame()
            with zf.open(target) as fh:
                raw = fh.read()
    except Exception:
        return pd.DataFrame()

    for kwargs in (
        dict(nrows=nrows),
        dict(sep=None, engine="python", nrows=nrows),
        dict(sep=";", engine="python", nrows=nrows),
        dict(sep="|", engine="python", nrows=nrows),
        dict(sep=None, engine="python", encoding="utf-8-sig", nrows=nrows),
        dict(sep=None, engine="python", encoding="latin1", nrows=nrows),
    ):
        try:
            buffer = io.BytesIO(raw)
            df = pd.read_csv(buffer, **kwargs)
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


def enrich_with_all_features(
    symbol: str,
    tf: str,
    start_ms: int,
    end_ms: int,
    root_dir: str,
    *,
    args: Optional[Any] = None,
    base_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Full enrichment used by both selector21 and the DL heads."""

    rule = _tf_to_rule(tf)
    sym = symbol.upper().strip()
    t0 = _to_utc(start_ms, unit="ms")
    t1 = _to_utc(end_ms, unit="ms")

    # ============================== OHLCV ===============================
    base_df = None
    if isinstance(base_data, dict):
        base_df = base_data.get(tf)
    dfs: list[pd.DataFrame] = []

    if base_df is not None and hasattr(base_df, "empty") and not base_df.empty:
        data = base_df.copy()
        if "time" not in data.columns and "open_time" in data.columns:
            data["time"] = data["open_time"]
        data["time"] = _to_utc(pd.to_numeric(data["time"], errors="coerce"), unit="ms")
        for col in ("open", "high", "low", "close", "volume"):
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors="coerce")
        dfs.append(
            data[[col for col in ("time", "open", "high", "low", "close", "volume") if col in data.columns]]
        )
    else:
        kl_dir_candidates = []
        if args is not None and getattr(args, "data_dir", ""):
            kl_dir_candidates.append(os.path.join(args.data_dir, "klines", tf))
        kl_dir_candidates.append(os.path.join(root_dir, "klines", tf))
        kl_dir_candidates.append(os.path.join("/opt/botscalp/datafull/klines", tf))

        kl_files: list[str] = []
        for directory in kl_dir_candidates:
            if os.path.isdir(directory):
                # BTCUSDT-5m-YYYY... ou BTCUSDT_5m_YYYY..., no diretório raiz
                found = sorted(glob.glob(os.path.join(directory, f"{sym}-{tf}-*.parquet")))
                if not found:
                    found = sorted(glob.glob(os.path.join(directory, f"{sym}_{tf}_*.parquet")))
                # Se ainda vazio, procurar na subpasta do símbolo
                if not found:
                    found = sorted(glob.glob(os.path.join(directory, sym, f"{sym}-{tf}-*.parquet")))
                if not found:
                    found = sorted(glob.glob(os.path.join(directory, sym, f"{sym}_{tf}_*.parquet")))
                if found:
                    print(f"[ENRICH][INFO] usando klines em: {directory} ({len(found)} arquivos)")
                    kl_files = found
                    break
        if not kl_files:
            raise FileNotFoundError(f"[ENRICH] Nenhum kline encontrado em {kl_dir_candidates}")

        for path in kl_files:
            data = pd.read_parquet(path)
            if data.empty:
                continue
            data.columns = [str(col).strip() for col in data.columns]
            if all(col.isdigit() for col in data.columns):
                cols = [
                    "time",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "close_time",
                    "quote_vol",
                    "n_trades",
                    "taker_buy_vol",
                    "taker_buy_quote_vol",
                    "ignore",
                ]
                data.columns = cols[: len(data.columns)]
            if "time" not in data.columns:
                if "open_time" in data.columns:
                    data["time"] = pd.to_numeric(data["open_time"], errors="coerce")
                else:
                    raise ValueError("[ENRICH] Klines sem coluna de tempo reconhecível")
            data["time"] = _to_utc(pd.to_numeric(data["time"], errors="coerce"), unit="ms")
            for col in ("open", "high", "low", "close", "volume"):
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors="coerce")
            dfs.append(
                data[[col for col in ("time", "open", "high", "low", "close", "volume") if col in data.columns]]
            )

    df = pd.concat(dfs, ignore_index=True)
    df = (
        df.dropna(subset=["time"])  # type: ignore[arg-type]
        .sort_values("time")
        .drop_duplicates(subset=["time"])
        .reset_index(drop=True)
    )
    df = df[(df["time"] >= t0) & (df["time"] <= t1)]

    grouped = df.set_index("time").resample(rule, label="right", closed="right")
    df = pd.DataFrame(
        {
            "time": grouped["close"].last().index,
            "open": grouped["open"].first(),
            "high": grouped["high"].max(),
            "low": grouped["low"].min(),
            "close": grouped["close"].last(),
            "volume": grouped["volume"].sum(),
        }
    ).dropna().reset_index(drop=True)

    # ======================= ATR regime =======================
    try:
        import talib as ta  # type: ignore
    except ImportError:
        print("[WARN] TA-Lib não disponível, ATR_zscore será calculado manualmente.")
        ta = None

    if all(col in df.columns for col in ["high", "low", "close"]):
        if ta:
            df["atr14"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=14)
            df["atr100"] = ta.ATR(df["high"], df["low"], df["close"], timeperiod=100)
        else:
            hl = df["high"] - df["low"]
            hc = (df["high"] - df["close"].shift()).abs()
            lc = (df["low"] - df["close"].shift()).abs()
            tr = pd.concat([hl, hc, lc], axis=1).max(axis=1)
            df["atr14"] = tr.rolling(14, min_periods=1).mean()
            df["atr100"] = tr.rolling(100, min_periods=1).mean()

        df["atr_zscore"] = df["atr14"] / (df["atr100"] + 1e-9)
        df["atr_zscore"] = df["atr_zscore"].replace([np.inf, -np.inf], np.nan).fillna(1.0)
        df["vol_bin"] = (df["atr_zscore"] > 1.3).astype(np.int8)

        mean_z = df["atr_zscore"].mean()
        high_vol_frac = (df["vol_bin"].sum() / len(df)) * 100.0
        print(f"[ENRICH][ATR_Z] Média z={mean_z:.2f} | Vol. alta em {high_vol_frac:.1f}% das barras.")
    else:
        print("[ENRICH][ATR_Z] Colunas high/low/close ausentes — ignorando ATR regime.")

    # ======================= Indicadores base =======================
    close = df["close"].astype(float)
    df["ema_fast"] = _ema(close, 12)
    df["ema_slow"] = _ema(close, 26)
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["rsi14"] = _rsi(close, 14)
    df["boll_pos"] = _boll_pos(close, 20, 2.0)
    df["atr_pct"] = _atr(df, 14) / (close.abs() + 1e-9)
    df["vwap20"] = (
        (df["close"] * df["volume"]).rolling(20).sum()
        / (df["volume"].rolling(20).sum() + 1e-9)
    )

    # ======================= AggTrades =======================
    agg_dir_candidates = []
    if args is not None and getattr(args, "agg_dir", ""):
        agg_dir_candidates.append(args.agg_dir)
    agg_dir_candidates.append(os.path.join(root_dir, "aggTrades"))
    agg_dir_candidates.append("/opt/botscalp/datafull/aggTrades")
    agg_dir_use = next((d for d in agg_dir_candidates if d and os.path.isdir(d)), None)

    if not agg_dir_use:
        print("[ENRICH][WARN] Nenhum diretório válido para aggTrades entre:", agg_dir_candidates)
        df["cvd_slope_agg"] = 0.0
    else:
        print(f"[ENRICH][INFO] Usando aggTrades em: {agg_dir_use}")
        agg_files = sorted(
            [
                os.path.join(agg_dir_use, file)
                for file in os.listdir(agg_dir_use)
                if file.startswith(f"{sym}-aggTrades-")
                and file.lower().endswith((".parquet", ".csv", ".zip"))
            ]
        )
        agg_frames = []
        for path in agg_files:
            try:
                if path.lower().endswith(".parquet"):
                    agg = pd.read_parquet(path)
                elif path.lower().endswith(".zip"):
                    with zipfile.ZipFile(path) as zf:
                        inner = zf.namelist()[0]
                        with zf.open(inner) as fh:
                            agg = pd.read_csv(fh, header=None)
                else:
                    agg = pd.read_csv(path, header=None)

                agg = agg.loc[:, ~agg.columns.duplicated()]
                if all(str(col).isdigit() for col in agg.columns) and agg.shape[1] >= 7:
                    agg = agg.iloc[:, :7]
                    agg.columns = [
                        "trade_id",
                        "price",
                        "qty",
                        "first_id",
                        "last_id",
                        "time",
                        "is_buyer_maker",
                    ]

                if "time" not in agg.columns:
                    cand = [
                        col
                        for col in agg.columns
                        if "time" in str(col).lower() or str(col).lower() in ("timestamp", "open_time")
                    ]
                    if cand:
                        agg = agg.rename(columns={cand[0]: "time"})
                    else:
                        continue

                agg["time"] = pd.to_numeric(agg["time"], errors="coerce")
                if agg["time"].max() < 1e12:
                    agg["time"] *= 1000
                agg["time"] = _to_utc(agg["time"].astype("int64"), unit="ms")

                agg = agg[(agg["time"] >= t0) & (agg["time"] <= t1)]
                if agg.empty:
                    continue

                agg["price"] = pd.to_numeric(agg.get("price"), errors="coerce")
                agg["qty"] = pd.to_numeric(agg.get("qty"), errors="coerce").fillna(0.0)
                agg["is_buyer_maker"] = agg.get("is_buyer_maker", False)
                agg["is_buyer_maker"] = (
                    agg["is_buyer_maker"].astype(str).str.lower().isin(["true", "1", "t", "yes"])
                )

                idx = agg.set_index("time").sort_index()
                idx["_notional"] = idx["price"] * idx["qty"]
                idx["_signed_notional"] = idx["_notional"] * np.where(idx["is_buyer_maker"], -1.0, +1.0)

                net = idx["_signed_notional"].resample(rule, label="right", closed="right").sum()
                buy = (
                    idx.loc[~idx["is_buyer_maker"], "_notional"].resample(rule, label="right", closed="right").sum()
                )
                sell = (
                    idx.loc[idx["is_buyer_maker"], "_notional"].resample(rule, label="right", closed="right").sum()
                )

                out = pd.DataFrame(
                    {
                        "time": net.index,
                        "cvd_slope_agg": net.values,
                        "flow_buy": buy.reindex(net.index).values,
                        "flow_sell": sell.reindex(net.index).values,
                    }
                ).reset_index(drop=True)
                agg_frames.append(out)
            except Exception as exc:
                print(f"[ENRICH][SKIP AGG] {os.path.basename(path)} -> {exc}")

        if agg_frames:
            agg_feat = pd.concat(agg_frames, ignore_index=True).dropna(how="all")
            df = df.merge(agg_feat, on="time", how="left")

    if "cvd_slope_agg" not in df.columns:
        df["cvd_slope_agg"] = 0.0
    cvd_src = df.get("cvd_slope_agg", 0.0)
    if not isinstance(cvd_src, pd.Series):
        cvd_src = pd.Series(cvd_src, index=df.index)
    df["cvd"] = cvd_src.fillna(0.0).cumsum()

    # ======================= BookDepth =======================
    depth_dir_candidates = []
    if args is not None and getattr(args, "depth_dir", ""):
        depth_dir_candidates.append(args.depth_dir)
    depth_dir_candidates.append(os.path.join(root_dir, "bookDepth"))
    depth_dir_candidates.append("/opt/botscalp/datafull/bookDepth")
    depth_dir_use = next((d for d in depth_dir_candidates if d and os.path.isdir(d)), None)

    if not depth_dir_use:
        print("[ENRICH][WARN] Nenhum diretório válido para bookDepth entre:", depth_dir_candidates)
        df["imb_net_depth"] = 0.0
    else:
        depth_files = sorted(
            [
                os.path.join(depth_dir_use, file)
                for file in os.listdir(depth_dir_use)
                if file.startswith(f"{sym}-bookDepth-")
                and file.lower().endswith((".parquet", ".csv", ".zip"))
            ]
        )
        depth_frames = []
        for path in depth_files:
            try:
                if path.lower().endswith(".parquet"):
                    depth = pd.read_parquet(path)
                elif path.lower().endswith(".zip"):
                    with zipfile.ZipFile(path) as zf:
                        inner = zf.namelist()[0]
                        with zf.open(inner) as fh:
                            depth = pd.read_csv(fh)
                else:
                    depth = pd.read_csv(path)

                depth = depth.loc[:, ~depth.columns.duplicated()]

                if {"timestamp", "percentage", "depth"} <= set(depth.columns):
                    ts = pd.to_numeric(depth["timestamp"], errors="coerce")
                    if ts.max() < 1e12:
                        ts = ts * 1000
                    depth["_time"] = _to_utc(ts.astype("int64"), unit="ms")
                    depth = depth[(depth["_time"] >= t0) & (depth["_time"] <= t1)]
                    if depth.empty:
                        continue

                    mask_pos = depth["percentage"] > 0
                    mask_neg = depth["percentage"] < 0
                    pos = (
                        pd.to_numeric(depth.loc[mask_pos, "depth"], errors="coerce")
                        .groupby(depth.loc[mask_pos, "_time"])
                        .sum()
                    )
                    neg = (
                        pd.to_numeric(depth.loc[mask_neg, "depth"], errors="coerce")
                        .groupby(depth.loc[mask_neg, "_time"])
                        .sum()
                    )
                    idx = pos.index.union(neg.index).sort_values()
                    pos = pos.reindex(idx, fill_value=0.0)
                    neg = neg.reindex(idx, fill_value=0.0)
                    tmp = pd.DataFrame(
                        {
                            "time": idx,
                            "imb_net_depth": (pos - neg) / (pos + neg + 1e-9),
                        }
                    )
                else:
                    column = "imb_net_depth"
                    if column not in depth.columns:
                        cand = [c for c in depth.columns if str(c).startswith("bd_imb_")]
                        if cand:
                            column = cand[0]

                    if "time" not in depth.columns:
                        candt = [
                            c
                            for c in depth.columns
                            if "time" in c.lower() or c.lower() == "timestamp"
                        ]
                        if candt:
                            depth = depth.rename(columns={candt[0]: "time"})
                    ts = pd.to_numeric(depth.get("time"), errors="coerce")
                    if ts.notna().any():
                        if ts.max() < 1e12:
                            ts = ts * 1000
                        depth["_time"] = _to_utc(ts.astype("int64"), unit="ms")
                        depth = depth[(depth["_time"] >= t0) & (depth["_time"] <= t1)]
                        tmp = pd.DataFrame(
                            {
                                "time": depth["_time"],
                                "imb_net_depth": pd.to_numeric(depth[column], errors="coerce"),
                            }
                        )
                    else:
                        tmp = pd.DataFrame(columns=["time", "imb_net_depth"])

                if not tmp.empty:
                    resampled = (
                        tmp.set_index("time").resample(rule, label="right", closed="right")["imb_net_depth"].mean()
                    )
                    depth_frames.append(resampled.reset_index())
            except Exception as exc:
                print(f"[ENRICH][SKIP DEPTH] {os.path.basename(path)} -> {exc}")

        if depth_frames:
            depth_feat = pd.concat(depth_frames, ignore_index=True)
            df = df.merge(depth_feat, on="time", how="left")

    if "imb_net_depth" not in df.columns:
        df["imb_net_depth"] = 0.0
    df["imb_ratio_depth"] = df["imb_net_depth"]

    # ======================= BookTicker =======================
    bbo_dir = os.path.join(root_dir, "bookTicker")
    bbo_paths = sorted(glob.glob(os.path.join(bbo_dir, f"{sym}*.parquet")))
    if bbo_paths:
        frames = []
        for path in bbo_paths:
            try:
                book = pd.read_parquet(path)
                book = book.loc[:, ~book.columns.duplicated()]
                book["time"] = _to_utc(pd.to_numeric(book["time"], errors="coerce"), unit="ms")
                book = book[(book["time"] >= t0) & (book["time"] <= t1)]
                if book.empty:
                    continue
                frames.append(book)
            except Exception as exc:
                print(f"[ENRICH][SKIP BBO] {os.path.basename(path)} -> {exc}")
        if frames:
            bbo = pd.concat(frames, ignore_index=True)
            keep = [c for c in bbo.columns if c in {"time", "bidPrice", "askPrice", "bidQty", "askQty"}]
            bbo = bbo[keep]
            bbo = bbo.drop_duplicates(subset=["time"]).sort_values("time")
            df = df.merge(bbo, on="time", how="left")
            df["mid"] = (pd.to_numeric(df.get("bidPrice"), errors="coerce") + pd.to_numeric(df.get("askPrice"), errors="coerce")) / 2
            df["spread"] = (pd.to_numeric(df.get("askPrice"), errors="coerce") - pd.to_numeric(df.get("bidPrice"), errors="coerce"))
            df["microprice_imb"] = (
                (pd.to_numeric(df.get("askPrice"), errors="coerce") * pd.to_numeric(df.get("bidQty"), errors="coerce"))
                - (pd.to_numeric(df.get("bidPrice"), errors="coerce") * pd.to_numeric(df.get("askQty"), errors="coerce"))
            ) / (
                (pd.to_numeric(df.get("askQty"), errors="coerce") + pd.to_numeric(df.get("bidQty"), errors="coerce"))
                + 1e-9
            )

    # ======================= Funding/Premium =======================
    for subdir, col in (
        ("fundingRate", "fundingRate"),
        ("premiumIndex", "premiumIndex"),
        ("markPriceKlines", "markPrice"),
    ):
        full_dir = os.path.join(root_dir, subdir)
        if not os.path.isdir(full_dir):
            continue
        paths = sorted(glob.glob(os.path.join(full_dir, f"{sym}-{subdir}-*.parquet")))
        if not paths:
            continue
        frames = []
        for path in paths:
            try:
                data = pd.read_parquet(path)
                data = data.loc[:, ~data.columns.duplicated()]
                time_col = "time" if "time" in data.columns else None
                if time_col is None:
                    cand = [c for c in data.columns if "time" in c.lower()]
                    if cand:
                        time_col = cand[0]
                if not time_col:
                    continue
                ts = pd.to_numeric(data[time_col], errors="coerce")
                if ts.max() < 1e12:
                    ts *= 1000
                data["time"] = _to_utc(ts.astype("int64"), unit="ms")
                data = data[(data["time"] >= t0) & (data["time"] <= t1)]
                if data.empty:
                    continue
                frames.append(data[["time", col]].rename(columns={col: col}))
            except Exception as exc:
                print(f"[ENRICH][SKIP {subdir.upper()}] {os.path.basename(path)} -> {exc}")
        if frames:
            feat = pd.concat(frames, ignore_index=True)
            resampled = feat.set_index("time").resample(rule, label="right", closed="right").last()
            df = df.merge(resampled.reset_index(), on="time", how="left")
            if col == "markPrice":
                df["mark_close_ratio"] = (df["markPrice"] / df["close"].replace(0, np.nan)) - 1.0

    # ======================= Metrics =======================
    metrics_dir = os.path.join(root_dir, "metrics")
    if os.path.isdir(metrics_dir):
        files = sorted(
            [
                os.path.join(metrics_dir, name)
                for name in os.listdir(metrics_dir)
                if name.startswith(f"{sym}-metrics-")
                and name.lower().endswith((".parquet", ".csv", ".zip"))
            ]
        )

        metric_frames = []
        for path in files:
            try:
                if path.lower().endswith(".parquet"):
                    table = pd.read_parquet(path)
                elif path.lower().endswith(".zip"):
                    table = _read_zip_any_csv(path)
                else:
                    table = None
                    for kwargs in (
                        dict(),
                        dict(sep=None, engine="python"),
                        dict(sep=";", engine="python"),
                        dict(sep="|", engine="python"),
                        dict(sep=None, engine="python", encoding="utf-8-sig"),
                        dict(sep=None, engine="python", encoding="latin1"),
                    ):
                        try:
                            table = pd.read_csv(path, **kwargs)
                            if table is not None and not table.empty:
                                break
                        except Exception:
                            table = None
                    if table is None:
                        table = pd.DataFrame()
                if table is None or table.empty:
                    print("[ENRICH][SKIP METRICS] vazio:", os.path.basename(path))
                    continue

                table = table.loc[:, ~table.columns.duplicated()]
                time_col = _pick_metrics_time_col(table)
                if not time_col:
                    print(
                        "[ENRICH][SKIP METRICS] sem coluna de tempo:",
                        os.path.basename(path),
                        "cols=",
                        list(table.columns)[:12],
                    )
                    continue

                ts = pd.to_numeric(table[time_col], errors="coerce")
                if ts.notna().sum() >= max(1, int(0.5 * len(table))):
                    if ts.max() < 1e12:
                        ts = ts * 1000
                    table["_time"] = pd.to_datetime(ts.astype("int64"), unit="ms", utc=True)
                else:
                    table["_time"] = pd.to_datetime(table[time_col], errors="coerce", utc=True)

                table = table[(table["_time"] >= t0) & (table["_time"] <= t1)]
                if table.empty:
                    continue

                def safe_div(num: str, den: str) -> pd.Series:
                    numerator = pd.to_numeric(table.get(num), errors="coerce")
                    denominator = pd.to_numeric(table.get(den), errors="coerce").replace(0, pd.NA)
                    return (numerator / denominator).astype("float64")

                feat = pd.DataFrame({"_time": table["_time"]})

                if "sum_open_interest" in table.columns:
                    feat["mx_oi"] = pd.to_numeric(table["sum_open_interest"], errors="coerce")
                if "sum_open_interest_value" in table.columns:
                    feat["mx_oi_value"] = pd.to_numeric(table["sum_open_interest_value"], errors="coerce")
                if {"sum_toptrader_long_short_ratio", "count_toptrader_long_short_ratio"} <= set(table.columns):
                    feat["mx_toptrader_lsr"] = safe_div(
                        "sum_toptrader_long_short_ratio", "count_toptrader_long_short_ratio"
                    )
                if {"sum_long_short_ratio", "count_long_short_ratio"} <= set(table.columns):
                    feat["mx_crowd_lsr"] = safe_div(
                        "sum_long_short_ratio", "count_long_short_ratio"
                    )
                if "sum_taker_long_short_vol_ratio" in table.columns:
                    feat["mx_taker_lsr_vol"] = pd.to_numeric(
                        table["sum_taker_long_short_vol_ratio"], errors="coerce"
                    )

                if feat.shape[1] == 1:
                    drops = {"_time", "symbol", "pair", "contract", "asset"}
                    for col in table.columns:
                        if col not in drops and pd.api.types.is_numeric_dtype(table[col]):
                            feat[f"mx_{col}"] = pd.to_numeric(table[col], errors="coerce")

                resampled = feat.set_index("_time").resample(rule, label="right", closed="right").mean()
                resampled = resampled.reset_index().rename(columns={"_time": "time"})
                metric_frames.append(resampled)
            except Exception as exc:
                print(f"[ENRICH][SKIP METRICS] {os.path.basename(path)} -> {exc}")

        if metric_frames:
            metric_feat = pd.concat(metric_frames, ignore_index=True)
            df = df.merge(metric_feat, on="time", how="left")

    # ======================= Normalizações & lags =======================
    if "cvd_slope_agg" in df.columns:
        base = (df["atr_pct"] * df["close"]).rolling(20).mean()
        df["cvd_norm"] = df["cvd_slope_agg"] / (base + 1e-9)
        df["cvd_slope_agg"] = _clip_std(df["cvd_slope_agg"])
    if "imb_net_depth" in df.columns:
        df["imb_zscore"] = (
            df["imb_net_depth"] - df["imb_net_depth"].rolling(50).mean()
        ) / (df["imb_net_depth"].rolling(50).std() + 1e-9)

    lag_cols = [
        "cvd_slope_agg",
        "cvd_norm",
        "imb_net_depth",
        "imb_ratio_depth",
        "imb_zscore",
        "fundingRate",
        "premiumIndex",
        "rsi14",
        "macd",
        "boll_pos",
        "mark_close_ratio",
    ]
    for lag in (1, 3, 5):
        for col in lag_cols:
            if col in df.columns:
                df[f"{col}_lag{lag}"] = df[col].shift(lag)

    df = df.sort_values("time").drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df["time_int"] = (df["time"].astype("int64") // 10**6).astype("int64")
    df = _downcast_inplace(df)

    if "cvd_slope_agg" not in df.columns:
        df["cvd_slope_agg"] = 0.0
    if "imb_net_depth" not in df.columns:
        df["imb_net_depth"] = 0.0
    return df


def enrich_from_request(request: EnrichmentRequest) -> pd.DataFrame:
    return enrich_with_all_features(
        request.symbol,
        request.timeframe,
        request.start_ms,
        request.end_ms,
        request.root_dir,
        args=request.args,
        base_data=request.base_data,
    )


def compute_feature_presence(df: pd.DataFrame) -> Dict[str, float | bool]:
    cvd_series = df.get("cvd_slope_agg", df.get("cvd"))
    if cvd_series is None:
        cvd_nonnull = 0.0
    else:
        cvd_nonnull = float(pd.Series(cvd_series).replace(0, np.nan).notna().mean())

    imb_series = df.get("imb_net_depth")
    if imb_series is None:
        imb_nonnull = 0.0
    else:
        imb_nonnull = float(pd.Series(imb_series).replace(0, np.nan).notna().mean())

    return {
        "cvd_present": "cvd" in df.columns or "cvd_slope_agg" in df.columns,
        "cvd_nonnull": cvd_nonnull,
        "imb_present": "imb_net_depth" in df.columns,
        "imb_nonnull": imb_nonnull,
        "bbo_present": "spread" in df.columns,
        "funding_present": "fundingRate" in df.columns,
        "premium_present": "premiumIndex" in df.columns,
        "mark_present": "markPrice" in df.columns,
    }


__all__ = [
    "EnrichmentRequest",
    "compute_feature_presence",
    "enrich_from_request",
    "enrich_with_all_features",
]
