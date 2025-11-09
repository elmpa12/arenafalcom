#!/usr/bin/env python3
"""
Treinar Modelo Deep Learning (LSTM)
Treina LSTM para filtrar sinais dos 4 setups validados
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import argparse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("🧠 Treinando Modelo LSTM para Trading\n")

class TradingDataset(Dataset):
    """Dataset para sequências de trading"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class TradingLSTM(nn.Module):
    """
    LSTM para classificação de trades
    Input: [batch, sequence_length, n_features]
    Output: [batch, 1] - probabilidade de trade bom (0-1)
    """
    def __init__(self, input_size, hidden_size=256, num_layers=2, dropout=0.3):
        super(TradingLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout)

        # Fully connected layers
        self.fc1 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [batch, seq_len, features]

        # LSTM 1
        lstm1_out, _ = self.lstm1(x)
        lstm1_out = self.dropout1(lstm1_out)

        # LSTM 2
        lstm2_out, _ = self.lstm2(lstm1_out)

        # Pegar último timestep
        last_timestep = lstm2_out[:, -1, :]
        last_timestep = self.dropout2(last_timestep)

        # Fully connected
        fc1_out = self.relu(self.fc1(last_timestep))
        fc1_out = self.dropout3(fc1_out)
        output = self.sigmoid(self.fc2(fc1_out))

        return output

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Treina uma época"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        # Forward pass
        optimizer.zero_grad()
        output = model(data)

        # Calculate loss
        loss = criterion(output.squeeze(), target)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predicted = (output.squeeze() > 0.5).float()
        correct += (predicted == target).sum().item()
        total += target.size(0)

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    """Valida o modelo"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output.squeeze(), target)

            total_loss += loss.item()
            predicted = (output.squeeze() > 0.5).float()
            correct += (predicted == target).sum().item()
            total += target.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total

    # Calcular precisão, recall, F1
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    tp = ((all_preds == 1) & (all_targets == 1)).sum()
    fp = ((all_preds == 1) & (all_targets == 0)).sum()
    fn = ((all_preds == 0) & (all_targets == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return avg_loss, accuracy, precision, recall, f1

def main():
    parser = argparse.ArgumentParser(description='Treinar LSTM para trading')
    parser.add_argument('--data_dir', default='./dl_data', help='Diretório com dataset')
    parser.add_argument('--hidden_size', type=int, default=256, help='Hidden size LSTM')
    parser.add_argument('--dropout', type=float, default=0.3, help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Número de epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--out_dir', default='./dl_models', help='Diretório de saída')

    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}\n")
    else:
        print("   (Sem GPU, usando CPU)\n")

    # Carregar dados
    data_dir = Path(args.data_dir)
    print(f"📂 Carregando dataset de {data_dir}...\n")

    X = np.load(data_dir / 'X_sequences.npy')
    y = np.load(data_dir / 'y_labels.npy')

    with open(data_dir / 'metadata.json') as f:
        metadata = json.load(f)

    print(f"  ✅ X shape: {X.shape}")
    print(f"  ✅ y shape: {y.shape}")
    print(f"  📊 Sequências: {len(X):,}")
    print(f"  📊 Features: {X.shape[2]}")
    print(f"  📊 Lookback: {X.shape[1]} barras\n")

    # Criar dataset e split
    dataset = TradingDataset(X, y)

    val_size = int(len(dataset) * args.val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"📊 Train set: {len(train_dataset):,} sequências")
    print(f"📊 Val set: {len(val_dataset):,} sequências ({args.val_split*100:.0f}%)\n")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device.type == 'cuda')
    )

    # Criar modelo
    input_size = X.shape[2]
    model = TradingLSTM(
        input_size=input_size,
        hidden_size=args.hidden_size,
        dropout=args.dropout
    ).to(device)

    print(f"🧠 Modelo criado:")
    print(f"   Input size: {input_size}")
    print(f"   Hidden size: {args.hidden_size}")
    print(f"   Dropout: {args.dropout}")
    print(f"   Parâmetros: {sum(p.numel() for p in model.parameters()):,}\n")

    # Loss e optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("="*70)
    print("🚀 INICIANDO TREINAMENTO\n")

    best_f1 = 0
    best_epoch = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_acc, val_prec, val_rec, val_f1 = validate(model, val_loader, criterion, device)

        # Log
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:5.1f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:5.1f}% "
              f"Prec: {val_prec:.3f} Rec: {val_rec:.3f} F1: {val_f1:.3f}")

        # Salvar histórico
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_prec,
            'val_recall': val_rec,
            'val_f1': val_f1
        })

        # Salvar melhor modelo (baseado em F1)
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_epoch = epoch

            out_dir = Path(args.out_dir)
            out_dir.mkdir(exist_ok=True)

            # Salvar modelo
            model_path = out_dir / 'trading_lstm_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': val_f1,
                'val_acc': val_acc,
                'input_size': input_size,
                'hidden_size': args.hidden_size,
                'dropout': args.dropout
            }, model_path)

    print("\n" + "="*70)
    print("✅ TREINAMENTO COMPLETO!\n")
    print(f"🏆 Melhor F1: {best_f1:.3f} (Epoch {best_epoch})\n")

    # Salvar histórico
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"💾 Modelo salvo em: {out_dir / 'trading_lstm_best.pth'}")
    print(f"📊 Histórico salvo em: {out_dir / 'training_history.json'}\n")

    # Estatísticas finais
    final_metrics = history[best_epoch - 1]
    print("📊 MÉTRICAS FINAIS (Melhor Epoch):\n")
    print(f"   Accuracy: {final_metrics['val_acc']:.1f}%")
    print(f"   Precision: {final_metrics['val_precision']:.3f}")
    print(f"   Recall: {final_metrics['val_recall']:.3f}")
    print(f"   F1-Score: {final_metrics['val_f1']:.3f}\n")

    if final_metrics['val_acc'] >= 80:
        print("🎯 ✅ META ATINGIDA! Accuracy >= 80%")
    elif final_metrics['val_acc'] >= 75:
        print("⚡ Accuracy >= 75% - BOM resultado!")
    else:
        print("⚠️  Accuracy < 75% - Considere ajustar hiperparâmetros")

    print("\n📊 Próximo passo:")
    print("   python3 validar_dl.py --model_dir ./dl_models\n")

if __name__ == '__main__':
    main()
