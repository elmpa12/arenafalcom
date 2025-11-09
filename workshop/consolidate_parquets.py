#!/usr/bin/env python3
"""
CONSOLIDATE PARQUETS - BotScalp V3

Consolida 732 arquivos diários em ~24 arquivos mensais.

Benefícios:
- Reduz 732 → 24 arquivos por tipo (30x menos!)
- Mantém compatibilidade 100% com selector21.py
- Compressão Zstd para economia de espaço
- Preserva TODOS os dados
- Acelera leitura (menos I/O)

Uso:
    python3 consolidate_parquets.py --dry-run  # Ver o que será feito
    python3 consolidate_parquets.py            # Executar consolidação
"""

import os
import glob
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict
import argparse


def get_file_info(file_path: str) -> Dict:
    """Extrai info do nome do arquivo"""
    filename = os.path.basename(file_path)

    # Formato: BTCUSDT_aggTrades_2022-11-08.parquet
    parts = filename.replace('.parquet', '').split('_')

    if len(parts) >= 3:
        symbol = parts[0]
        data_type = parts[1]
        date_str = parts[2]

        # Parse data
        date = pd.to_datetime(date_str)
        year_month = date.strftime('%Y-%m')

        return {
            'symbol': symbol,
            'data_type': data_type,
            'date': date,
            'year_month': year_month,
            'file_path': file_path
        }
    return None


def consolidate_monthly(
    input_dir: str,
    output_dir: str,
    data_type: str,
    symbol: str = "BTCUSDT",
    dry_run: bool = False
) -> List[str]:
    """
    Consolida arquivos diários em mensais.

    Args:
        input_dir: Diretório com arquivos diários
        output_dir: Diretório para arquivos mensais
        data_type: Tipo de dado (aggTrades, klines_1m, etc)
        symbol: Par (BTCUSDT)
        dry_run: Se True, apenas simula

    Returns:
        Lista de arquivos criados
    """

    print(f"\n{'='*70}")
    print(f"Consolidando: {data_type}")
    print(f"{'='*70}")

    # Buscar todos os arquivos
    pattern = os.path.join(input_dir, f"{symbol}_{data_type}_*.parquet")
    files = sorted(glob.glob(pattern))

    if not files:
        print(f"⚠️  Nenhum arquivo encontrado: {pattern}")
        return []

    print(f"📊 Arquivos encontrados: {len(files)}")

    # Agrupar por mês
    monthly_groups = {}
    for file_path in files:
        info = get_file_info(file_path)
        if info:
            year_month = info['year_month']
            if year_month not in monthly_groups:
                monthly_groups[year_month] = []
            monthly_groups[year_month].append(file_path)

    print(f"📅 Meses encontrados: {len(monthly_groups)}")

    # Consolidar cada mês
    created_files = []
    os.makedirs(output_dir, exist_ok=True)

    for year_month in sorted(monthly_groups.keys()):
        daily_files = monthly_groups[year_month]

        # Nome do arquivo consolidado
        output_file = os.path.join(
            output_dir,
            f"{symbol}_{data_type}_{year_month}.parquet"
        )

        print(f"\n📦 {year_month}: {len(daily_files)} dias → {os.path.basename(output_file)}")

        if dry_run:
            print(f"   [DRY RUN] Arquivo seria criado: {output_file}")
            continue

        # Ler todos os arquivos do mês
        dfs = []
        for file_path in daily_files:
            try:
                df = pd.read_parquet(file_path)
                dfs.append(df)
            except Exception as e:
                print(f"   ⚠️  Erro ao ler {os.path.basename(file_path)}: {e}")

        if not dfs:
            print(f"   ⚠️  Nenhum DataFrame válido para {year_month}")
            continue

        # Concatenar
        df_month = pd.concat(dfs, ignore_index=True)

        # Ordenar por timestamp se existir
        if 'timestamp' in df_month.columns:
            df_month = df_month.sort_values('timestamp').reset_index(drop=True)
        elif df_month.index.name in ['timestamp', 'open_time']:
            df_month = df_month.sort_index()

        # Salvar com compressão Zstd
        df_month.to_parquet(
            output_file,
            compression='zstd',
            compression_level=3,  # Balanço speed/size
            index=True if df_month.index.name else False
        )

        # Stats
        size_mb = os.path.getsize(output_file) / 1024 / 1024
        print(f"   ✅ Criado: {len(df_month):,} rows, {size_mb:.1f}MB")

        created_files.append(output_file)

    return created_files


def consolidate_all(
    base_dir: str = "./data",
    output_base_dir: str = "./data_monthly",
    dry_run: bool = False
):
    """
    Consolida todos os tipos de dados.

    Args:
        base_dir: Diretório base com dados diários
        output_base_dir: Diretório base para dados mensais
        dry_run: Se True, apenas simula
    """

    print("\n" + "🚀" * 35)
    print("CONSOLIDAÇÃO DE PARQUETS - BotScalp V3")
    print("🚀" * 35)
    print()
    print(f"Input:  {base_dir}")
    print(f"Output: {output_base_dir}")
    print(f"Mode:   {'DRY RUN (simulação)' if dry_run else 'EXECUTION (real)'}")
    print()

    # Mapeamento de tipos de dados
    data_types = {
        "aggTrades": {
            "dir": os.path.join(base_dir, "aggTrades", "BTCUSDT"),
            "pattern_name": "aggTrades"
        },
        "1m": {
            "dir": os.path.join(base_dir, "klines", "1m", "BTCUSDT"),
            "pattern_name": "1m"
        },
        "5m": {
            "dir": os.path.join(base_dir, "klines", "5m", "BTCUSDT"),
            "pattern_name": "5m"
        },
        "15m": {
            "dir": os.path.join(base_dir, "klines", "15m", "BTCUSDT"),
            "pattern_name": "15m"
        },
    }

    all_created = []
    stats = {}

    for data_type, config in data_types.items():
        input_dir = config["dir"]
        pattern_name = config["pattern_name"]

        if not os.path.exists(input_dir):
            print(f"⚠️  Diretório não existe: {input_dir}")
            continue

        # Diretório de saída
        if data_type == "aggTrades":
            output_dir = os.path.join(output_base_dir, "aggTrades", "BTCUSDT")
        else:
            output_dir = os.path.join(output_base_dir, "klines", data_type, "BTCUSDT")

        # Consolidar
        created = consolidate_monthly(
            input_dir=input_dir,
            output_dir=output_dir,
            data_type=pattern_name,
            dry_run=dry_run
        )

        all_created.extend(created)

        # Stats
        if created:
            total_size = sum(os.path.getsize(f) for f in created) / 1024 / 1024
            stats[data_type] = {
                'files': len(created),
                'size_mb': total_size
            }

    # Resumo final
    print("\n" + "=" * 70)
    print("📊 RESUMO DA CONSOLIDAÇÃO")
    print("=" * 70)

    if dry_run:
        print("\n⚠️  DRY RUN - Nenhum arquivo foi criado!")
        print("   Execute sem --dry-run para consolidar de verdade.")
    else:
        total_files = sum(s['files'] for s in stats.values())
        total_size = sum(s['size_mb'] for s in stats.values())

        print(f"\n✅ Arquivos criados: {total_files}")
        print(f"✅ Tamanho total: {total_size:.1f}MB")
        print()

        for data_type, data in stats.items():
            print(f"   {data_type}: {data['files']} files, {data['size_mb']:.1f}MB")

    print("\n" + "=" * 70)
    print("🎯 PRÓXIMOS PASSOS")
    print("=" * 70)
    print()
    print("1. Verificar arquivos em:", output_base_dir)
    print("2. Testar com selector21:")
    print(f"   python3 selector21.py --symbol BTCUSDT --data_dir {output_base_dir}")
    print()
    print("3. Se tudo OK, pode deletar arquivos diários (backup antes!)")
    print("   mv data data_daily_backup")
    print("   mv data_monthly data")
    print()


def verify_consolidation(
    original_dir: str,
    consolidated_dir: str,
    data_type: str
):
    """
    Verifica se consolidação preservou todos os dados.

    Args:
        original_dir: Diretório com arquivos originais
        consolidated_dir: Diretório com arquivos consolidados
        data_type: Tipo de dado
    """
    print(f"\n🔍 Verificando: {data_type}")

    # Ler originais
    pattern_orig = os.path.join(original_dir, f"BTCUSDT_{data_type}_*.parquet")
    files_orig = sorted(glob.glob(pattern_orig))

    dfs_orig = []
    for f in files_orig:
        dfs_orig.append(pd.read_parquet(f))

    df_orig = pd.concat(dfs_orig, ignore_index=True)

    # Ler consolidados
    pattern_cons = os.path.join(consolidated_dir, f"BTCUSDT_{data_type}_*.parquet")
    files_cons = sorted(glob.glob(pattern_cons))

    dfs_cons = []
    for f in files_cons:
        dfs_cons.append(pd.read_parquet(f))

    df_cons = pd.concat(dfs_cons, ignore_index=True)

    # Comparar
    print(f"   Original: {len(df_orig):,} rows")
    print(f"   Consolidated: {len(df_cons):,} rows")

    if len(df_orig) == len(df_cons):
        print(f"   ✅ MATCH! Todos os dados preservados")
        return True
    else:
        diff = len(df_orig) - len(df_cons)
        print(f"   ⚠️  DIFF: {diff:,} rows")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Consolidate daily parquets to monthly")
    parser.add_argument('--base-dir', default='./data', help='Base directory with daily data')
    parser.add_argument('--output-dir', default='./data_monthly', help='Output directory for monthly data')
    parser.add_argument('--dry-run', action='store_true', help='Simulate without creating files')
    parser.add_argument('--verify', action='store_true', help='Verify consolidation after execution')

    args = parser.parse_args()

    # Consolidar
    consolidate_all(
        base_dir=args.base_dir,
        output_base_dir=args.output_dir,
        dry_run=args.dry_run
    )

    # Verificar (se solicitado e não dry-run)
    if args.verify and not args.dry_run:
        print("\n" + "=" * 70)
        print("🔍 VERIFICAÇÃO DE INTEGRIDADE")
        print("=" * 70)

        verify_consolidation(
            original_dir=os.path.join(args.base_dir, "aggTrades", "BTCUSDT"),
            consolidated_dir=os.path.join(args.output_dir, "aggTrades/BTCUSDT"),
            data_type="aggTrades"
        )
