# risk_manager.py
from dataclasses import dataclass

@dataclass
class RiskLimits:
  hard_stop_brl: float = -3000.0
  soft_stop_brl: float = -2000.0
  cooldown_losses: int = 3

@dataclass
class DayState:
  pnl_brl: float = 0.0
  losses_streak: int = 0
  hard_stop_hit: bool = False
  soft_stop_hit: bool = False
  cooldown: bool = False

class RiskManager:
  def __init__(self, limits: RiskLimits):
    self.limits = limits
    self.state = DayState()

  def on_trade_closed(self, pl_brl: float):
    self.state.pnl_brl += pl_brl
    if pl_brl < 0:
      self.state.losses_streak += 1
    else:
      self.state.losses_streak = 0
    self._refresh_flags()

  def _refresh_flags(self):
    L, S = self.limits, self.state
    S.hard_stop_hit = (S.pnl_brl <= L.hard_stop_brl)
    S.soft_stop_hit = (S.pnl_brl <= L.soft_stop_brl)
    S.cooldown      = (S.losses_streak >= L.cooldown_losses)

  def allow_new_trade(self, tier: str) -> bool:
    # tier: '15m' | '5m' | '1m'
    if self.state.hard_stop_hit:
      return False
    if self.state.cooldown and tier in ('1m','5m'):
      return False
    if self.state.soft_stop_hit and tier in ('1m','5m'):
      return False
    return True
