#!/usr/bin/env python3
"""
Validar Modelo Deep Learning
Testa performance do LSTM em dados de teste
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🧪 Validando Modelo LSTM\n")

class TradingLSTM(nn.Module):
    """LSTM para classificação de trades (mesma arquitetura do treino)"""
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.3):
        super(TradingLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0)
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(hidden_size, 128, num_layers=1, batch_first=True, dropout=0)
        self.dropout2 = nn.Dropout(dropout)

        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        lstm2_out, _ = self.lstm2(lstm1_out)
        last_timestep = lstm2_out[:, -1, :]
        last_timestep = self.dropout2(last_timestep)

        fc1_out = self.relu(self.fc1(last_timestep))
        fc1_out = self.dropout3(fc1_out)
        output = self.sigmoid(self.fc2(fc1_out))

        return output

def evaluate_model(model, X, y, device, confidence_threshold=0.7):
    """
    Avalia modelo em dados de teste
    Retorna métricas detalhadas
    """
    model.eval()

    X_tensor = torch.FloatTensor(X).to(device)

    with torch.no_grad():
        outputs = model(X_tensor)
        confidences = outputs.squeeze().cpu().numpy()

    # Predições (threshold 0.5)
    predictions = (confidences > 0.5).astype(int)

    # Métricas básicas
    accuracy = (predictions == y).mean()

    tp = ((predictions == 1) & (y == 1)).sum()
    fp = ((predictions == 1) & (y == 0)).sum()
    fn = ((predictions == 0) & (y == 1)).sum()
    tn = ((predictions == 0) & (y == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Métricas com filtro de confidence
    high_confidence_mask = confidences > confidence_threshold
    n_filtered = high_confidence_mask.sum()
    filter_rate = n_filtered / len(confidences) * 100

    if n_filtered > 0:
        filtered_predictions = predictions[high_confidence_mask]
        filtered_targets = y[high_confidence_mask]

        acc_filtered = (filtered_predictions == filtered_targets).mean()

        tp_f = ((filtered_predictions == 1) & (filtered_targets == 1)).sum()
        fp_f = ((filtered_predictions == 1) & (filtered_targets == 0)).sum()
        fn_f = ((filtered_predictions == 0) & (filtered_targets == 1)).sum()

        prec_filtered = tp_f / (tp_f + fp_f) if (tp_f + fp_f) > 0 else 0
        rec_filtered = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0
        f1_filtered = 2 * (prec_filtered * rec_filtered) / (prec_filtered + rec_filtered) if (prec_filtered + rec_filtered) > 0 else 0
    else:
        acc_filtered = prec_filtered = rec_filtered = f1_filtered = 0

    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'fn': int(fn),
        'tn': int(tn),
        'n_samples': len(y),
        'confidence_threshold': confidence_threshold,
        'n_high_confidence': int(n_filtered),
        'filter_rate': float(filter_rate),
        'accuracy_filtered': float(acc_filtered),
        'precision_filtered': float(prec_filtered),
        'recall_filtered': float(rec_filtered),
        'f1_filtered': float(f1_filtered)
    }

    return results, confidences

def main():
    parser = argparse.ArgumentParser(description='Validar LSTM treinado')
    parser.add_argument('--model_dir', default='./dl_models', help='Diretório com modelo treinado')
    parser.add_argument('--data_dir', default='./dl_data', help='Diretório com dataset')
    parser.add_argument('--test_split', type=float, default=0.2, help='% para teste (últimos dados)')
    parser.add_argument('--confidence_threshold', type=float, default=0.7, help='Threshold de confidence')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}\n")

    # Carregar dados
    data_dir = Path(args.data_dir)
    print(f"📂 Carregando dataset de {data_dir}...\n")

    X = np.load(data_dir / 'X_sequences.npy')
    y = np.load(data_dir / 'y_labels.npy')

    # Split: últimos N% para teste (dados mais recentes)
    test_size = int(len(X) * args.test_split)
    X_test = X[-test_size:]
    y_test = y[-test_size:]

    print(f"  ✅ Test set: {len(X_test):,} sequências ({args.test_split*100:.0f}% dos dados)")
    print(f"  📊 Features: {X_test.shape[2]}")
    print(f"  📊 Lookback: {X_test.shape[1]} barras\n")

    # Carregar modelo
    model_dir = Path(args.model_dir)
    model_path = model_dir / 'trading_lstm_best.pth'

    if not model_path.exists():
        print(f"❌ Modelo não encontrado: {model_path}")
        print("   Execute treinar_dl.py primeiro!\n")
        return

    print(f"📥 Carregando modelo de {model_path}...\n")

    checkpoint = torch.load(model_path, map_location=device)

    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    dropout = checkpoint['dropout']

    model = TradingLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        dropout=dropout
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"  ✅ Modelo carregado (Epoch {checkpoint['epoch']})")
    print(f"  📊 Train F1: {checkpoint['val_f1']:.3f}")
    print(f"  📊 Train Acc: {checkpoint['val_acc']:.1f}%\n")

    print("="*70)
    print("🧪 TESTANDO NO TEST SET\n")

    # Avaliar
    results, confidences = evaluate_model(
        model, X_test, y_test, device,
        confidence_threshold=args.confidence_threshold
    )

    # Mostrar resultados
    print("📊 MÉTRICAS GERAIS (threshold 0.5):\n")
    print(f"   Accuracy: {results['accuracy']*100:.1f}%")
    print(f"   Precision: {results['precision']:.3f}")
    print(f"   Recall: {results['recall']:.3f}")
    print(f"   F1-Score: {results['f1_score']:.3f}\n")

    print(f"   TP: {results['tp']:,} | FP: {results['fp']:,}")
    print(f"   FN: {results['fn']:,} | TN: {results['tn']:,}\n")

    print("="*70)
    print(f"🎯 FILTRO DE CONFIDENCE (threshold {args.confidence_threshold}):\n")

    print(f"   Sinais mantidos: {results['n_high_confidence']:,} / {results['n_samples']:,} ({results['filter_rate']:.1f}%)")
    print(f"   Sinais filtrados: {results['n_samples'] - results['n_high_confidence']:,} ({100-results['filter_rate']:.1f}%)\n")

    if results['n_high_confidence'] > 0:
        print(f"   Accuracy (filtrado): {results['accuracy_filtered']*100:.1f}%")
        print(f"   Precision (filtrado): {results['precision_filtered']:.3f}")
        print(f"   Recall (filtrado): {results['recall_filtered']:.3f}")
        print(f"   F1-Score (filtrado): {results['f1_filtered']:.3f}\n")

        # Comparação
        acc_gain = (results['accuracy_filtered'] - results['accuracy']) * 100
        print(f"   📈 Ganho de Accuracy: {acc_gain:+.1f}%")

        if results['accuracy_filtered'] >= 0.80:
            print(f"   🎯 ✅ META ATINGIDA! Accuracy >= 80% após filtro")
        elif results['accuracy_filtered'] >= 0.75:
            print(f"   ⚡ Accuracy >= 75% - BOM resultado!")
        else:
            print(f"   ⚠️  Accuracy < 75% - Considere retreinar ou ajustar threshold")

    # Salvar resultados
    results['tested_at'] = datetime.now().isoformat()
    results['model_path'] = str(model_path)
    results['test_set_size'] = int(test_size)

    out_file = model_dir / 'validation_results.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Resultados salvos em: {out_file}\n")

    print("="*70)
    print("\n💡 PRÓXIMO PASSO:\n")
    print("   Integrar modelo DL com selector21 para filtrar sinais dos 4 setups!")
    print("   Criar script: integrar_dl_selector.py\n")

if __name__ == '__main__':
    main()
