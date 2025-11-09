#!/usr/bin/env python3
"""
TESTES ULTRA-RÁPIDOS - Feedback Máximo

Cada teste: 5-15 segundos
Objetivo: Aprendizado rápido com alta frequência de feedback

Uso:
    python3 ultra_fast_tests.py --batch_size 20
"""

# TESTES ULTRA-RÁPIDOS (1 semana, 1-2 métodos, 1 timeframe)
ULTRA_FAST_TESTS = [
    # Semana 1 - Jan 2024
    {"name": "rapid_w1_trend", "start": "2024-01-01", "end": "2024-01-08", "tf": "1m", "methods": "trend_breakout"},
    {"name": "rapid_w1_rsi", "start": "2024-01-01", "end": "2024-01-08", "tf": "1m", "methods": "rsi_reversion"},
    {"name": "rapid_w1_ema", "start": "2024-01-01", "end": "2024-01-08", "tf": "1m", "methods": "ema_crossover"},
    {"name": "rapid_w1_macd", "start": "2024-01-01", "end": "2024-01-08", "tf": "1m", "methods": "macd_trend"},
    {"name": "rapid_w1_vwap", "start": "2024-01-01", "end": "2024-01-08", "tf": "1m", "methods": "vwap_trend"},

    # Semana 2
    {"name": "rapid_w2_trend", "start": "2024-01-08", "end": "2024-01-15", "tf": "1m", "methods": "trend_breakout"},
    {"name": "rapid_w2_rsi", "start": "2024-01-08", "end": "2024-01-15", "tf": "1m", "methods": "rsi_reversion"},
    {"name": "rapid_w2_ema", "start": "2024-01-08", "end": "2024-01-15", "tf": "1m", "methods": "ema_crossover"},
    {"name": "rapid_w2_boll", "start": "2024-01-08", "end": "2024-01-15", "tf": "1m", "methods": "boll_breakout"},
    {"name": "rapid_w2_kelt", "start": "2024-01-08", "end": "2024-01-15", "tf": "1m", "methods": "keltner_breakout"},

    # Semana 3
    {"name": "rapid_w3_trend", "start": "2024-01-15", "end": "2024-01-22", "tf": "5m", "methods": "trend_breakout"},
    {"name": "rapid_w3_rsi", "start": "2024-01-15", "end": "2024-01-22", "tf": "5m", "methods": "rsi_reversion"},
    {"name": "rapid_w3_macd", "start": "2024-01-15", "end": "2024-01-22", "tf": "5m", "methods": "macd_trend"},
    {"name": "rapid_w3_orb", "start": "2024-01-15", "end": "2024-01-22", "tf": "5m", "methods": "orb_breakout"},
    {"name": "rapid_w3_don", "start": "2024-01-15", "end": "2024-01-22", "tf": "5m", "methods": "donchian_breakout"},

    # Semana 4
    {"name": "rapid_w4_ema", "start": "2024-01-22", "end": "2024-01-29", "tf": "15m", "methods": "ema_crossover"},
    {"name": "rapid_w4_macd", "start": "2024-01-22", "end": "2024-01-29", "tf": "15m", "methods": "macd_trend"},
    {"name": "rapid_w4_vwap", "start": "2024-01-22", "end": "2024-01-29", "tf": "15m", "methods": "vwap_trend"},
    {"name": "rapid_w4_emapull", "start": "2024-01-22", "end": "2024-01-29", "tf": "15m", "methods": "ema_pullback"},
    {"name": "rapid_w4_orr", "start": "2024-01-22", "end": "2024-01-29", "tf": "15m", "methods": "orr_reversal"},

    # Fevereiro - Volatilidade
    {"name": "rapid_feb_w1_trend", "start": "2024-02-01", "end": "2024-02-08", "tf": "1m", "methods": "trend_breakout"},
    {"name": "rapid_feb_w1_rsi", "start": "2024-02-01", "end": "2024-02-08", "tf": "1m", "methods": "rsi_reversion"},
    {"name": "rapid_feb_w2_macd", "start": "2024-02-08", "end": "2024-02-15", "tf": "5m", "methods": "macd_trend"},
    {"name": "rapid_feb_w2_vwap", "start": "2024-02-08", "end": "2024-02-15", "tf": "5m", "methods": "vwap_trend"},
    {"name": "rapid_feb_w3_boll", "start": "2024-02-15", "end": "2024-02-22", "tf": "15m", "methods": "boll_breakout"},

    # Março
    {"name": "rapid_mar_w1_trend", "start": "2024-03-01", "end": "2024-03-08", "tf": "1m", "methods": "trend_breakout"},
    {"name": "rapid_mar_w1_ema", "start": "2024-03-01", "end": "2024-03-08", "tf": "1m", "methods": "ema_crossover"},
    {"name": "rapid_mar_w2_macd", "start": "2024-03-08", "end": "2024-03-15", "tf": "5m", "methods": "macd_trend"},
    {"name": "rapid_mar_w2_rsi", "start": "2024-03-08", "end": "2024-03-15", "tf": "5m", "methods": "rsi_reversion"},
    {"name": "rapid_mar_w3_vwap", "start": "2024-03-15", "end": "2024-03-22", "tf": "15m", "methods": "vwap_trend"},

    # Combos ultra-rápidos
    {"name": "rapid_combo_w1", "start": "2024-01-01", "end": "2024-01-08", "tf": "1m", "methods": "trend_breakout,rsi_reversion", "combos": True},
    {"name": "rapid_combo_w2", "start": "2024-01-08", "end": "2024-01-15", "tf": "5m", "methods": "ema_crossover,macd_trend", "combos": True},
    {"name": "rapid_combo_w3", "start": "2024-01-15", "end": "2024-01-22", "tf": "15m", "methods": "vwap_trend,boll_breakout", "combos": True},
]


def generate_test_config(test_spec, out_dir):
    """Gera config de teste a partir do spec."""
    args = [
        "--umcsv_root", "./data_monthly",
        "--symbol", "BTCUSDT",
        "--start", test_spec["start"],
        "--end", test_spec["end"],
        "--interval", "auto",
        "--exec_rules", test_spec["tf"],
        "--methods", test_spec["methods"],
        "--run_base",
        "--n_jobs", "2",  # Apenas 2 cores por teste (permite +paralelos)
        "--par_backend", "thread",
        "--out_root", f"{out_dir}/{test_spec['name']}",
    ]

    if test_spec.get("combos"):
        args.extend([
            "--run_combos",
            "--combo_ops", "AND",
            "--combo_cap", "5",  # Apenas 5 combos (ultra-rápido)
        ])

    return {
        "name": test_spec["name"],
        "desc": f"{test_spec['tf']} {test_spec['methods']} {test_spec['start'][:10]}",
        "args": args
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=20, help="Testes por batch")
    parser.add_argument("--out_dir", type=str, default="./resultados/rapid", help="Diretório output")
    args = parser.parse_args()

    # Gera configs
    configs = []
    for spec in ULTRA_FAST_TESTS[:args.batch_size]:
        configs.append(generate_test_config(spec, args.out_dir))

    # Salva
    with open("ultra_fast_tests_config.json", "w") as f:
        json.dump(configs, f, indent=2)

    print(f"✓ Gerados {len(configs)} testes ultra-rápidos")
    print(f"  Cada teste: ~5-15 segundos")
    print(f"  Total estimado: {len(configs)*10}s = {len(configs)*10/60:.1f}min com paralização")
    print(f"\nPróximo passo:")
    print(f"  python3 run_from_config.py ultra_fast_tests_config.json --parallel {args.batch_size//2}")
