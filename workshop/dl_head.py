"""Módulo de treino leve e explícito para o backend do selector."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from backend.pipeline import DeepLearningDataset, DeepLearningResult


@dataclass
class DLHeadConfig:
    """Hiperparâmetros básicos para um head denso binário."""

    hidden_sizes: Sequence[int] = (128, 64)
    dropout: float = 0.1
    lr: float = 1e-3
    max_epochs: int = 30
    batch_size: int = 2048
    patience: int = 5
    device: str = "auto"  # "cpu", "cuda" ou "auto"
    threshold: Optional[float] = None
    threshold_grid: Sequence[float] = field(
        default_factory=lambda: (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
    )
    clip_grad_norm: Optional[float] = 5.0
    weight_decay: float = 1e-4


class _FeedForwardHead(nn.Module):
    def __init__(self, in_features: int, config: DLHeadConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last = in_features
        for hidden in config.hidden_sizes:
            layers.append(nn.Linear(last, hidden))
            layers.append(nn.ReLU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            last = hidden
        layers.append(nn.Linear(last, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return self.model(x)


class DeepLearningHead:
    """Treina um classificador binário simples e coeso para o selector."""

    def __init__(self, config: Optional[DLHeadConfig] = None) -> None:
        self.config = config or DLHeadConfig()
        self.device = self._resolve_device()
        self._model: Optional[_FeedForwardHead] = None
        self._norm_stats: Dict[str, Dict[str, float]] | None = None

    def _resolve_device(self) -> torch.device:
        if self.config.device != "auto":
            return torch.device(self.config.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, dataset: DeepLearningDataset) -> DeepLearningResult:
        feature_columns = dataset.feature_columns
        if not feature_columns:
            raise ValueError("Dataset sem colunas numéricas para treino.")

        train_frame = dataset.train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        val_frame = dataset.validation.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if train_frame.empty or val_frame.empty:
            raise ValueError("Dataset sem dados suficientes para treino/validação.")

        train_X = train_frame[feature_columns].astype(np.float32)
        val_X = val_frame[feature_columns].astype(np.float32)

        mean = train_X.mean()
        std = train_X.std(ddof=0).replace(0, 1.0)
        self._norm_stats = {"mean": mean.to_dict(), "std": std.to_dict()}

        train_tensor = torch.from_numpy(((train_X - mean) / std).to_numpy())
        val_tensor = torch.from_numpy(((val_X - mean) / std).to_numpy())

        train_y = torch.from_numpy(train_frame[dataset.label_column].astype(np.float32).to_numpy())
        val_y = torch.from_numpy(val_frame[dataset.label_column].astype(np.float32).to_numpy())

        train_w = self._weights_from_frame(dataset, train_frame)
        val_w = self._weights_from_frame(dataset, val_frame)

        train_data = TensorDataset(train_tensor, train_y, train_w)
        loader = DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True, drop_last=False)

        model = _FeedForwardHead(len(feature_columns), self.config).to(self.device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        criterion = nn.BCEWithLogitsLoss(reduction="none")

        best_state = None
        best_val_loss = float("inf")
        epochs_without_improve = 0

        val_tensor = val_tensor.to(self.device)
        val_y = val_y.to(self.device)
        val_w = val_w.to(self.device)

        for epoch in range(self.config.max_epochs):
            model.train()
            for batch_X, batch_y, batch_w in loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_w = batch_w.to(self.device)

                optimizer.zero_grad(set_to_none=True)
                logits = model(batch_X).squeeze(-1)
                loss = criterion(logits, batch_y)
                loss = (loss * batch_w).mean()
                loss.backward()
                if self.config.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip_grad_norm)
                optimizer.step()

            val_loss = self._evaluate_loss(model, criterion, val_tensor, val_y, val_w)
            if val_loss + 1e-6 < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improve = 0
            else:
                epochs_without_improve += 1

            if epochs_without_improve >= self.config.patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model

        train_probs = self._predict_proba(model, train_tensor.to(self.device)).cpu().numpy()
        val_probs = self._predict_proba(model, val_tensor).cpu().numpy()

        train_returns = train_frame[dataset.return_column].to_numpy(dtype=np.float32)
        val_returns = val_frame[dataset.return_column].to_numpy(dtype=np.float32)

        train_labels = train_frame[dataset.label_column].to_numpy(dtype=np.float32)
        val_labels = val_frame[dataset.label_column].to_numpy(dtype=np.float32)

        threshold = self._select_threshold(val_probs, val_returns)
        metrics = {
            "train": self._summary_metrics(train_probs, train_labels, train_returns, threshold),
            "validation": self._summary_metrics(val_probs, val_labels, val_returns, threshold),
        }

        predictions = self._build_predictions(dataset, train_probs, val_probs, threshold)
        model_state = {
            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "config": asdict(self.config),
            "normalization": self._norm_stats,
        }

        return DeepLearningResult(
            timeframe=dataset.timeframe,
            feature_columns=feature_columns,
            metrics=metrics,
            predictions=predictions,
            model_state=model_state,
            best_threshold=threshold,
            metadata=dataset.metadata,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _weights_from_frame(self, dataset: DeepLearningDataset, frame: pd.DataFrame) -> torch.Tensor:
        if dataset.weight_column and dataset.weight_column in frame:
            weights = frame[dataset.weight_column].astype(np.float32).to_numpy()
            weights = np.clip(weights, 1e-6, None)
        else:
            weights = np.ones(len(frame), dtype=np.float32)
        return torch.from_numpy(weights)

    def _evaluate_loss(
        self,
        model: nn.Module,
        criterion: nn.Module,
        features: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor,
    ) -> float:
        model.eval()
        with torch.no_grad():
            logits = model(features).squeeze(-1)
            loss = criterion(logits, labels)
            return float((loss * weights).mean().cpu().item())

    def _predict_proba(self, model: nn.Module, features: torch.Tensor) -> torch.Tensor:
        model.eval()
        with torch.no_grad():
            logits = model(features).squeeze(-1)
            return torch.sigmoid(logits)

    def _select_threshold(self, probs: np.ndarray, future_returns: np.ndarray) -> float:
        if self.config.threshold is not None:
            return float(self.config.threshold)
        best_thr, best_score = 0.5, -np.inf
        for thr in self.config.threshold_grid:
            pnl = float(np.mean(np.where(probs >= thr, future_returns, 0.0)))
            if pnl > best_score:
                best_score = pnl
                best_thr = float(thr)
        return best_thr

    def _summary_metrics(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        future_returns: np.ndarray,
        threshold: float,
    ) -> Dict[str, float]:
        preds = (probs >= threshold).astype(np.float32)
        coverage = float(preds.mean())
        if coverage <= 0:
            avg_ret = 0.0
            hit_rate = 0.0
        else:
            selected_returns = future_returns[preds > 0]
            avg_ret = float(selected_returns.mean()) if len(selected_returns) else 0.0
            hits = labels[preds > 0]
            hit_rate = float(hits.mean()) if len(hits) else 0.0
        accuracy = float((preds == labels).mean()) if len(labels) else 0.0
        return {
            "accuracy": accuracy,
            "coverage": coverage,
            "avg_return": avg_ret,
            "hit_rate": hit_rate,
        }

    def _build_predictions(
        self,
        dataset: DeepLearningDataset,
        train_probs: np.ndarray,
        val_probs: np.ndarray,
        threshold: float,
    ) -> pd.DataFrame:
        def _pack(frame: pd.DataFrame, probs: np.ndarray, split: str) -> pd.DataFrame:
            out = frame[["time", dataset.return_column, dataset.label_column]].copy()
            out["prob"] = probs
            out["split"] = split
            out["decision"] = (probs >= threshold).astype(np.float32)
            return out

        frames = [
            _pack(dataset.train, train_probs, "train"),
            _pack(dataset.validation, val_probs, "validation"),
        ]
        return pd.concat(frames, ignore_index=True)
