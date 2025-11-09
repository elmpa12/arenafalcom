#!/usr/bin/env python3
"""
Model Signal Generator - Gera sinais de trading baseados em modelos treinados

Extrai os melhores modelos do Walk-Forward e usa para predictions em tempo real:
- Carrega modelos ML (XGBoost, RF, LogisticRegression) salvos
- Carrega modelos DL (GRU, TCN) se disponíveis
- Gera sinais BUY/SELL/HOLD com score de confiança
- Integra múltiplos timeframes e métodos
"""

import os
import sys
import json
import glob
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
import numpy as np
import pandas as pd

try:
    import joblib
except ImportError:
    print("⚠️  Installing joblib...")
    os.system("pip3 install -q joblib")
    import joblib


@dataclass
class TradingSignal:
    """Sinal de trading gerado pelo modelo"""
    timestamp: str
    symbol: str
    signal: str  # "BUY", "SELL", "HOLD"
    confidence: float  # 0.0 to 1.0
    method: str  # "ml_xgb", "dl_gru", "combo", etc
    timeframe: str  # "1m", "5m", "15m"
    features: Dict  # Features usadas na decisão
    raw_prediction: float  # Valor bruto do modelo

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'signal': self.signal,
            'confidence': self.confidence,
            'method': self.method,
            'timeframe': self.timeframe,
            'features': self.features,
            'raw_prediction': self.raw_prediction
        }


class ModelSignalGenerator:
    """
    Gerador de sinais usando modelos treinados do selector21 e dl_heads
    """

    def __init__(
        self,
        models_dir: str = "./ml_models",
        symbol: str = "BTCUSDT",
        timeframes: List[str] = ["1m", "5m", "15m"]
    ):
        """
        Initialize signal generator

        Args:
            models_dir: Diretório com modelos salvos
            symbol: Par de trading
            timeframes: Lista de timeframes
        """
        self.models_dir = Path(models_dir)
        self.symbol = symbol
        self.timeframes = timeframes

        self.ml_models = {}  # {timeframe: {method: (model, scaler)}}
        self.dl_models = {}  # {timeframe: model}

        print(f"🔍 Carregando modelos de: {self.models_dir}")
        self._load_best_models()

    def _load_best_models(self):
        """Carrega os melhores modelos treinados"""

        if not self.models_dir.exists():
            print(f"⚠️  Diretório {self.models_dir} não existe!")
            print(f"   Execute primeiro: python3 selector21.py --run_ml --ml_save_dir {self.models_dir}")
            return

        # Carrega modelos ML (selector21)
        ml_files = list(self.models_dir.glob("model_*.pkl"))
        scaler_files = list(self.models_dir.glob("scaler_*.pkl"))

        print(f"   Encontrados: {len(ml_files)} modelos ML, {len(scaler_files)} scalers")

        for model_file in ml_files:
            try:
                # Parse nome: model_SYMBOL_1m_xgb_wf0.pkl
                parts = model_file.stem.split('_')
                if len(parts) < 4:
                    continue

                symbol = parts[1]
                if symbol != self.symbol:
                    continue

                timeframe = parts[2]
                method = parts[3]

                # Carrega modelo
                model = joblib.load(model_file)

                # Procura scaler correspondente
                scaler_file = self.models_dir / f"scaler_{model_file.stem.replace('model_', '')}.pkl"
                scaler = None
                if scaler_file.exists():
                    scaler = joblib.load(scaler_file)

                # Armazena
                if timeframe not in self.ml_models:
                    self.ml_models[timeframe] = {}

                self.ml_models[timeframe][method] = (model, scaler)
                print(f"   ✅ Loaded: {timeframe} {method}")

            except Exception as e:
                print(f"   ⚠️  Erro ao carregar {model_file.name}: {e}")

        # TODO: Carrega modelos DL (dl_heads_v8)
        # dl_files = list(self.models_dir.glob("*.h5"))
        # ...

        print(f"\n✅ Modelos carregados:")
        for tf, methods in self.ml_models.items():
            print(f"   {tf}: {list(methods.keys())}")

    def get_latest_market_data(self, timeframe: str = "1m") -> pd.DataFrame:
        """
        Obtém dados de mercado mais recentes

        Na produção, isso viria de:
        - Binance API (klines)
        - WebSocket stream
        - Arquivo parquet local atualizado

        Por enquanto, simula com dados históricos
        """
        # TODO: Integrar com Binance API ou dados reais
        # Por enquanto, retorna DataFrame vazio com schema correto

        df = pd.DataFrame({
            'timestamp': [datetime.now()],
            'open': [95000.0],
            'high': [95100.0],
            'low': [94900.0],
            'close': [95050.0],
            'volume': [1000.0],
            'rsi': [55.0],
            'macd': [10.0],
            'macd_signal': [8.0],
            'bb_upper': [96000.0],
            'bb_lower': [94000.0],
            'atr': [150.0]
        })

        return df

    def extract_features(self, df: pd.DataFrame, timeframe: str) -> Dict:
        """
        Extrai features do DataFrame para usar no modelo

        Deve replicar as mesmas features que foram usadas no treinamento!
        """
        if df.empty:
            return {}

        last_row = df.iloc[-1]

        features = {
            # Price features
            'close': last_row.get('close', 0),
            'high': last_row.get('high', 0),
            'low': last_row.get('low', 0),
            'volume': last_row.get('volume', 0),

            # Technical indicators
            'rsi': last_row.get('rsi', 50),
            'macd': last_row.get('macd', 0),
            'macd_signal': last_row.get('macd_signal', 0),
            'atr': last_row.get('atr', 0),

            # Derived
            'macd_diff': last_row.get('macd', 0) - last_row.get('macd_signal', 0),
            'price_range': last_row.get('high', 0) - last_row.get('low', 0),
        }

        return features

    def predict_with_ml_model(
        self,
        model,
        scaler,
        features: Dict,
        threshold: float = 0.5
    ) -> Tuple[str, float, float]:
        """
        Faz prediction com modelo ML

        Returns:
            (signal, confidence, raw_prediction)
        """
        # Converte features para array
        feature_names = sorted(features.keys())
        X = np.array([[features[k] for k in feature_names]])

        # Scale
        if scaler is not None:
            X = scaler.transform(X)

        # Predict
        try:
            if hasattr(model, 'predict_proba'):
                # Modelo com probabilidades (RF, LogReg, etc)
                proba = model.predict_proba(X)[0]
                if len(proba) == 2:
                    # Binary: [prob_0, prob_1]
                    prob_long = proba[1]
                else:
                    prob_long = proba.max()

                raw_prediction = prob_long

            else:
                # XGBoost ou modelo de regressão
                raw_prediction = float(model.predict(X)[0])
                prob_long = raw_prediction

            # Converte para sinal
            if prob_long > threshold + 0.1:  # Zona de alta confiança
                signal = "BUY"
                confidence = prob_long
            elif prob_long < threshold - 0.1:  # Zona de baixa confiança
                signal = "SELL"
                confidence = 1.0 - prob_long
            else:
                signal = "HOLD"
                confidence = 0.5  # Neutro

            return signal, confidence, raw_prediction

        except Exception as e:
            print(f"⚠️  Erro na prediction: {e}")
            return "HOLD", 0.5, 0.5

    def generate_signal(
        self,
        timeframe: str = "5m",
        method: str = "auto"
    ) -> Optional[TradingSignal]:
        """
        Gera sinal de trading para timeframe específico

        Args:
            timeframe: "1m", "5m", "15m", etc
            method: "xgb", "rf", "logreg", "auto" (usa melhor)

        Returns:
            TradingSignal ou None se não há modelos
        """
        # Verifica se há modelos para este timeframe
        if timeframe not in self.ml_models:
            print(f"⚠️  Nenhum modelo para {timeframe}")
            return None

        # Obtém dados de mercado
        df = self.get_latest_market_data(timeframe)
        features = self.extract_features(df, timeframe)

        if not features:
            print(f"⚠️  Não foi possível extrair features")
            return None

        # Seleciona método
        available_methods = list(self.ml_models[timeframe].keys())

        if method == "auto":
            # Usa primeiro disponível (em produção, escolher o melhor baseado em backtest)
            method = available_methods[0] if available_methods else None

        if method not in self.ml_models[timeframe]:
            print(f"⚠️  Método {method} não disponível. Disponíveis: {available_methods}")
            return None

        # Faz prediction
        model, scaler = self.ml_models[timeframe][method]
        signal, confidence, raw_pred = self.predict_with_ml_model(
            model, scaler, features
        )

        # Cria objeto TradingSignal
        trading_signal = TradingSignal(
            timestamp=datetime.now().isoformat(),
            symbol=self.symbol,
            signal=signal,
            confidence=confidence,
            method=f"ml_{method}",
            timeframe=timeframe,
            features=features,
            raw_prediction=raw_pred
        )

        return trading_signal

    def generate_multi_timeframe_signal(self) -> Dict[str, TradingSignal]:
        """
        Gera sinais para todos os timeframes configurados

        Returns:
            Dict {timeframe: signal}
        """
        signals = {}

        for tf in self.timeframes:
            signal = self.generate_signal(timeframe=tf, method="auto")
            if signal:
                signals[tf] = signal

        return signals

    def generate_consensus_signal(self) -> Optional[TradingSignal]:
        """
        Gera sinal de consenso entre múltiplos timeframes

        Estratégia:
        - Se todos concordam (BUY/SELL): alta confiança
        - Se maioria concorda: média confiança
        - Se dividido: HOLD
        """
        signals = self.generate_multi_timeframe_signal()

        if not signals:
            return None

        # Conta votos
        buy_count = sum(1 for s in signals.values() if s.signal == "BUY")
        sell_count = sum(1 for s in signals.values() if s.signal == "SELL")
        hold_count = sum(1 for s in signals.values() if s.signal == "HOLD")
        total = len(signals)

        # Calcula confiança média
        avg_confidence = np.mean([s.confidence for s in signals.values()])

        # Decisão
        if buy_count > sell_count and buy_count > hold_count:
            final_signal = "BUY"
            final_confidence = avg_confidence * (buy_count / total)
        elif sell_count > buy_count and sell_count > hold_count:
            final_signal = "SELL"
            final_confidence = avg_confidence * (sell_count / total)
        else:
            final_signal = "HOLD"
            final_confidence = 0.5

        # Cria sinal de consenso
        consensus = TradingSignal(
            timestamp=datetime.now().isoformat(),
            symbol=self.symbol,
            signal=final_signal,
            confidence=final_confidence,
            method="consensus",
            timeframe="multi",
            features={'buy_votes': buy_count, 'sell_votes': sell_count, 'hold_votes': hold_count},
            raw_prediction=avg_confidence
        )

        return consensus


def test_signal_generator():
    """Testa gerador de sinais"""
    print("="*70)
    print("🧪 TESTE - MODEL SIGNAL GENERATOR")
    print("="*70 + "\n")

    # Inicializa
    generator = ModelSignalGenerator(
        models_dir="./ml_models",
        symbol="BTCUSDT",
        timeframes=["1m", "5m", "15m"]
    )

    # Testa sinal único
    print("\n📊 Gerando sinal para 5m...")
    signal = generator.generate_signal(timeframe="5m", method="auto")

    if signal:
        print(f"\n✅ Sinal gerado:")
        print(f"   Signal: {signal.signal}")
        print(f"   Confidence: {signal.confidence:.2%}")
        print(f"   Method: {signal.method}")
        print(f"   Raw prediction: {signal.raw_prediction:.4f}")
    else:
        print("\n⚠️  Nenhum sinal gerado (modelos não encontrados)")
        print("\n💡 Para treinar modelos:")
        print("   python3 selector21.py --symbol BTCUSDT --run_ml --ml_save_dir ./ml_models")

    # Testa multi-timeframe
    print("\n📊 Gerando sinais multi-timeframe...")
    signals = generator.generate_multi_timeframe_signal()

    if signals:
        print(f"\n✅ Sinais por timeframe:")
        for tf, sig in signals.items():
            print(f"   {tf}: {sig.signal} (conf: {sig.confidence:.2%})")

    # Testa consenso
    print("\n📊 Gerando sinal de consenso...")
    consensus = generator.generate_consensus_signal()

    if consensus:
        print(f"\n✅ Consenso:")
        print(f"   Signal: {consensus.signal}")
        print(f"   Confidence: {consensus.confidence:.2%}")
        print(f"   Features: {consensus.features}")

    print("\n" + "="*70)
    print("✅ Teste concluído!")
    print("="*70)


if __name__ == "__main__":
    test_signal_generator()
