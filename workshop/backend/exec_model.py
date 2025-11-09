# exec_model.py
from dataclasses import dataclass
from typing import Literal, Dict

Side = Literal["long", "short"]

@dataclass
class FeesConfig:
    # B3 "tarifa única" (faixa base) e desconto de day-trade
    tarifa_base_pct: float = 0.000004  # 0,0004%%
    desconto_daytrade: float = 0.70    # 70%%

def b3_fee_per_side(price_brl: float, cfg: FeesConfig) -> float:
    taxa = cfg.tarifa_base_pct * price_brl
    return taxa * (1 - cfg.desconto_daytrade)

@dataclass
class SlipConfig:
    tick: float = 0.5           # tamanho do tick (BRL) – ajuste para o BIT
    alpha_spread: float = 0.5   # fração do spread
    beta_range: float = 0.10    # fração do range do candle
    stress_mult: float = 1.0    # multiplicador em picos (opcional)

def slip_price(side: Side, price: float, spread: float, bar_range: float, cfg: SlipConfig) -> float:
    slip = max(cfg.tick, cfg.alpha_spread * spread + cfg.beta_range * bar_range)
    return price + (slip if side == "long" else -slip)

def apply_execution(entry_price: float, side: Side, candle: Dict, cfg: SlipConfig) -> float:
    # candle: {open,high,low,close, spread?, range?}
    spread = candle.get("spread", max(0.0, candle["high"] - candle["low"]) * 0.05)
    bar_range = candle.get("range",  candle["high"] - candle["low"])
    return slip_price(side, entry_price, spread, bar_range, cfg)

def realized_pl(entry_px: float, exit_px: float, side: Side, qty: float = 1.0) -> float:
    return (exit_px - entry_px) * (1 if side == "long" else -1) * qty

def apply_fees(pl_brl: float, entry_px_brl: float, exit_px_brl: float, cfg: FeesConfig, qty: float = 1.0) -> float:
    fee_entry = b3_fee_per_side(entry_px_brl, cfg) * qty
    fee_exit  = b3_fee_per_side(exit_px_brl,  cfg) * qty
    return pl_brl - (fee_entry + fee_exit)
