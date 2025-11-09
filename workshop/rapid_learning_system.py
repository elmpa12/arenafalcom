#!/usr/bin/env python3
"""
RAPID LEARNING SYSTEM - Aprendizado em Tempo Real
Feedback instantâneo durante execução, sem esperar backtests terminarem
"""

import json
import subprocess
import time
import threading
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque

class RapidLearningSystem:
    """Sistema de aprendizado rápido com feedback em tempo real"""

    def __init__(self):
        self.session_id = datetime.now().strftime("%Y-%m-%d_%H%M")
        self.session_dir = Path(f"sessions/rapid_learning_{self.session_id}")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # METAS CLARAS
        self.TARGETS = {
            "hit_min": 0.48,      # Hit rate mínimo aceitável
            "payoff_min": 1.15,   # Payoff mínimo
            "maxdd_max": -3000,   # Max drawdown aceitável
            "sharpe_min": 0.2,    # Sharpe mínimo
            "pnl_min": 0          # PnL positivo
        }

        # Aprendizado em tempo real
        self.learnings = []
        self.running_tests = {}
        self.real_time_metrics = {}

        print(f"\n{'='*80}")
        print(f"⚡ RAPID LEARNING SYSTEM")
        print(f"{'='*80}")
        print(f"Session: {self.session_id}")
        print(f"\n🎯 METAS CLARAS:")
        for metric, target in self.TARGETS.items():
            print(f"   {metric}: {target}")
        print(f"{'='*80}\n")

    def generate_varied_periods(self):
        """Gera períodos em MESES DIFERENTES, nunca repetindo"""

        periods = []

        # Lista de meses disponíveis (2 anos de dados)
        available_months = [
            ("2022-07", "Jul 2022"), ("2022-08", "Ago 2022"), ("2022-09", "Set 2022"),
            ("2022-10", "Out 2022"), ("2022-11", "Nov 2022"), ("2022-12", "Dez 2022"),
            ("2023-01", "Jan 2023"), ("2023-02", "Fev 2023"), ("2023-03", "Mar 2023"),
            ("2023-04", "Abr 2023"), ("2023-05", "Mai 2023"), ("2023-06", "Jun 2023"),
            ("2023-07", "Jul 2023"), ("2023-08", "Ago 2023"), ("2023-09", "Set 2023"),
            ("2023-10", "Out 2023"), ("2023-11", "Nov 2023"), ("2023-12", "Dez 2023"),
            ("2024-01", "Jan 2024"), ("2024-02", "Fev 2024"), ("2024-03", "Mar 2024"),
            ("2024-04", "Abr 2024"), ("2024-05", "Mai 2024"), ("2024-06", "Jun 2024"),
        ]

        used_months = set()

        # 5 testes de 15 dias (em meses diferentes)
        for i in range(5):
            # Escolher mês não usado
            month_start, month_name = available_months[i * 4]  # Espaçar 4 meses
            used_months.add(month_start)

            start_date = f"{month_start}-01"
            end_date = f"{month_start}-15"

            periods.append({
                "name": f"15d_{month_name.replace(' ', '_')}",
                "start": start_date,
                "end": end_date,
                "days": 15,
                "month": month_name
            })

        # 10 testes de 5 dias (em meses diferentes dos anteriores)
        offset = 5 * 4
        for i in range(10):
            month_idx = (offset + i * 2) % len(available_months)
            month_start, month_name = available_months[month_idx]

            if month_start in used_months:
                month_idx = (month_idx + 1) % len(available_months)
                month_start, month_name = available_months[month_idx]

            used_months.add(month_start)

            # Variar dias do mês
            day_start = (i % 4) * 5 + 1  # 1, 6, 11, 16
            day_end = day_start + 5

            start_date = f"{month_start}-{day_start:02d}"
            end_date = f"{month_start}-{day_end:02d}"

            periods.append({
                "name": f"5d_{month_name.replace(' ', '_')}_d{day_start}",
                "start": start_date,
                "end": end_date,
                "days": 5,
                "month": month_name
            })

        # 5 testes de 30 dias (meses completos diferentes)
        for i in range(5):
            month_idx = (i * 5) % len(available_months)
            month_start, month_name = available_months[month_idx]

            if month_start in used_months:
                month_idx = (month_idx + 2) % len(available_months)
                month_start, month_name = available_months[month_idx]

            used_months.add(month_start)

            # Mês completo
            year, month = month_start.split('-')
            start_date = f"{month_start}-01"

            # Último dia do mês
            if month in ['01', '03', '05', '07', '08', '10', '12']:
                end_date = f"{month_start}-31"
            elif month == '02':
                end_date = f"{month_start}-28"
            else:
                end_date = f"{month_start}-30"

            periods.append({
                "name": f"30d_{month_name.replace(' ', '_')}",
                "start": start_date,
                "end": end_date,
                "days": 30,
                "month": month_name
            })

        return periods

    def create_test_batch(self, batch_name: str, periods: list, methods: list, tf: str = "5m"):
        """Cria batch de testes com períodos variados"""

        tests = []

        for i, period in enumerate(periods):
            # Variar métodos
            method = methods[i % len(methods)]

            test_config = {
                "name": f"{batch_name}_{i:02d}_{period['name']}",
                "period": period,
                "method": method,
                "tf": tf,
                "args": [
                    "--umcsv_root", "./data_monthly",
                    "--symbol", "BTCUSDT",
                    "--start", period["start"],
                    "--end", period["end"],
                    "--exec_rules", tf,
                    "--methods", method,
                    "--run_base",
                    "--n_jobs", "2",
                    "--out_root", str(self.session_dir / f"{batch_name}_{i:02d}")
                ],
                "targets": self.TARGETS.copy()
            }

            tests.append(test_config)

        return tests

    def monitor_test_realtime(self, test_name: str, test_dir: Path):
        """Monitora teste em TEMPO REAL via logs"""

        log_file = test_dir / "test.log"
        csv_file = test_dir / "leaderboard_base.csv"

        # Inicializar métricas
        self.real_time_metrics[test_name] = {
            "status": "running",
            "progress": 0,
            "alerts": []
        }

        # Thread para monitorar
        def monitor():
            last_size = 0

            while test_name in self.running_tests:
                time.sleep(2)  # Check a cada 2 segundos

                # Verificar progresso via log
                if log_file.exists():
                    current_size = log_file.stat().st_size

                    if current_size > last_size:
                        # Ler novas linhas
                        with open(log_file, 'r') as f:
                            f.seek(last_size)
                            new_lines = f.readlines()
                            last_size = current_size

                            # Procurar por indicadores de progresso
                            for line in new_lines:
                                # Exemplo: se log contém "Processing period..."
                                if "Processing" in line or "Backtest" in line:
                                    self.real_time_metrics[test_name]["progress"] += 5

                # Verificar se CSV já existe (teste concluído)
                if csv_file.exists():
                    metrics = self._read_csv_metrics(csv_file)

                    if metrics:
                        # ANÁLISE EM TEMPO REAL
                        alerts = self._analyze_metrics_realtime(test_name, metrics)

                        self.real_time_metrics[test_name]["metrics"] = metrics
                        self.real_time_metrics[test_name]["alerts"] = alerts
                        self.real_time_metrics[test_name]["status"] = "completed"

                        # APRENDIZADO IMEDIATO
                        self._learn_from_test(test_name, metrics, alerts)

                        break

        # Iniciar thread de monitoramento
        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()

    def _analyze_metrics_realtime(self, test_name: str, metrics: dict) -> list:
        """Analisa métricas EM TEMPO REAL e gera alertas"""

        alerts = []

        # Verificar cada meta
        if metrics["hit"] < self.TARGETS["hit_min"]:
            alerts.append({
                "type": "WARNING",
                "metric": "hit",
                "value": metrics["hit"],
                "target": self.TARGETS["hit_min"],
                "message": f"Hit rate abaixo do alvo: {metrics['hit']:.2%} < {self.TARGETS['hit_min']:.2%}",
                "action": "Considerar aumentar atr_stop_mult"
            })

        if metrics.get("payoff", 0) < self.TARGETS["payoff_min"]:
            alerts.append({
                "type": "WARNING",
                "metric": "payoff",
                "value": metrics.get("payoff", 0),
                "target": self.TARGETS["payoff_min"],
                "message": f"Payoff abaixo do alvo: {metrics.get('payoff', 0):.2f} < {self.TARGETS['payoff_min']}",
                "action": "Considerar aumentar hard_tp_usd"
            })

        if metrics["total_pnl"] >= 0:
            alerts.append({
                "type": "SUCCESS",
                "metric": "pnl",
                "value": metrics["total_pnl"],
                "message": f"✅ PnL POSITIVO: {metrics['total_pnl']:,.0f}",
                "action": "Salvar configuração como promissora"
            })

        if metrics["sharpe"] >= self.TARGETS["sharpe_min"]:
            alerts.append({
                "type": "SUCCESS",
                "metric": "sharpe",
                "value": metrics["sharpe"],
                "message": f"✅ Sharpe acima do alvo: {metrics['sharpe']:.2f}",
                "action": "Marcar para análise detalhada"
            })

        return alerts

    def _learn_from_test(self, test_name: str, metrics: dict, alerts: list):
        """APRENDIZADO IMEDIATO - registra insights"""

        learning = {
            "timestamp": datetime.now().isoformat(),
            "test": test_name,
            "metrics": metrics,
            "alerts": alerts,
            "insights": []
        }

        # Gerar insights
        if metrics["total_pnl"] > 0:
            learning["insights"].append(f"🟢 LUCRATIVO: {test_name} gerou +{metrics['total_pnl']:,.0f}")

        if metrics["sharpe"] > 1.0:
            learning["insights"].append(f"⭐ SHARPE ALTO: {metrics['sharpe']:.2f} - configuração muito promissora")

        if metrics["hit"] > 0.55:
            learning["insights"].append(f"🎯 HIT ALTO: {metrics['hit']:.2%} - método consistente")

        # Adicionar ao registro
        self.learnings.append(learning)

        # Print em tempo real
        print(f"\n{'─'*80}")
        print(f"⚡ APRENDIZADO EM TEMPO REAL - {test_name}")
        print(f"{'─'*80}")
        for insight in learning["insights"]:
            print(f"  {insight}")

        for alert in alerts:
            icon = "✅" if alert["type"] == "SUCCESS" else "⚠️"
            print(f"  {icon} {alert['message']}")

        print(f"{'─'*80}\n")

    def _read_csv_metrics(self, csv_path: Path):
        """Lê métricas do CSV"""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                row = df.iloc[0]
                return {
                    "hit": row["hit"],
                    "payoff": row.get("payoff", 0),
                    "total_pnl": row["total_pnl"],
                    "sharpe": row["sharpe"],
                    "maxdd": row["maxdd"],
                    "n_trades": row["n_trades"]
                }
        except Exception as e:
            return None

    def run_rapid_learning(self):
        """Executa sistema de aprendizado rápido"""

        # 1. Gerar períodos variados
        periods = self.generate_varied_periods()

        print(f"📅 Períodos gerados (meses diferentes):")
        for p in periods[:5]:
            print(f"   {p['name']}: {p['start']} a {p['end']} ({p['days']} dias)")
        print(f"   ... {len(periods)} períodos totais")

        # 2. Métodos baseados em Gen 3 (melhores descobertos)
        methods = [
            "macd_trend",      # +277K em Gen3
            "trend_breakout",  # +194K, Sharpe 1.13
            "ema_crossover",   # +160K
            "vwap_trend",
            "rsi_reversion"
        ]

        # 3. Criar batches
        # Batch 1: 5 testes de 15 dias
        batch1 = self.create_test_batch("batch1_15d", periods[:5], methods, tf="5m")

        # Batch 2: 10 testes de 5 dias
        batch2 = self.create_test_batch("batch2_5d", periods[5:15], methods, tf="5m")

        # Batch 3: 5 testes de 30 dias
        batch3 = self.create_test_batch("batch3_30d", periods[15:20], methods, tf="15m")  # 15m para 30 dias

        all_tests = batch1 + batch2 + batch3

        print(f"\n📊 Testes criados:")
        print(f"   Batch 1: {len(batch1)} testes (15 dias)")
        print(f"   Batch 2: {len(batch2)} testes (5 dias)")
        print(f"   Batch 3: {len(batch3)} testes (30 dias)")
        print(f"   TOTAL: {len(all_tests)} testes\n")

        # 4. Executar em paralelo com feedback em tempo real
        self._execute_with_realtime_feedback(all_tests, parallel=8)

        # 5. Relatório final
        self._generate_learning_report()

    def _execute_with_realtime_feedback(self, tests: list, parallel: int = 8):
        """Executa testes com feedback em tempo real"""

        running = []
        pending = list(tests)
        completed = []

        print(f"\n{'='*80}")
        print(f"🚀 EXECUTANDO COM FEEDBACK EM TEMPO REAL")
        print(f"{'='*80}")
        print(f"Total: {len(tests)} testes | Paralelo: {parallel}")
        print(f"{'='*80}\n")

        while pending or running:
            # Iniciar novos testes
            while len(running) < parallel and pending:
                test = pending.pop(0)

                # Criar diretório
                out_dir = Path(test["args"][test["args"].index("--out_root") + 1])
                out_dir.mkdir(parents=True, exist_ok=True)

                log_file = out_dir / "test.log"

                # Iniciar processo
                proc = subprocess.Popen(
                    ["python3", "selector21.py"] + test["args"],
                    stdout=open(log_file, "w"),
                    stderr=subprocess.STDOUT
                )

                test_info = {
                    "name": test["name"],
                    "proc": proc,
                    "start": time.time(),
                    "out_dir": out_dir,
                    "test_config": test
                }

                running.append(test_info)
                self.running_tests[test["name"]] = test_info

                # Iniciar monitoramento em tempo real
                self.monitor_test_realtime(test["name"], out_dir)

                print(f"[{len(completed)+len(running)}/{len(tests)}] ▶️  {test['name']} ({test['period']['days']}d, {test['method']})")

            # Verificar testes concluídos
            time.sleep(1)

            done = [r for r in running if r["proc"].poll() is not None]
            for r in done:
                elapsed = time.time() - r["start"]
                status = "✅" if r["proc"].returncode == 0 else "❌"

                # Remover do running
                running.remove(r)
                del self.running_tests[r["name"]]

                completed.append({
                    "name": r["name"],
                    "elapsed": elapsed,
                    "success": r["proc"].returncode == 0,
                    "real_time_data": self.real_time_metrics.get(r["name"])
                })

                print(f"{status} {r['name']} ({elapsed:.1f}s)")

        print(f"\n✅ Todos os {len(completed)} testes completados!\n")

        return completed

    def _generate_learning_report(self):
        """Gera relatório de aprendizados"""

        report_path = self.session_dir / "learning_report.md"

        with open(report_path, "w") as f:
            f.write(f"# Rapid Learning Report - {self.session_id}\n\n")

            # Resumo
            total_learnings = len(self.learnings)
            profitable = [l for l in self.learnings if l["metrics"]["total_pnl"] > 0]
            high_sharpe = [l for l in self.learnings if l["metrics"]["sharpe"] > 0.5]

            f.write(f"## Resumo Executivo\n\n")
            f.write(f"- Total de testes: {total_learnings}\n")
            f.write(f"- Lucrativos: {len(profitable)} ({len(profitable)/total_learnings*100:.1f}%)\n")
            f.write(f"- Alto Sharpe (>0.5): {len(high_sharpe)} ({len(high_sharpe)/total_learnings*100:.1f}%)\n\n")

            # Top performers
            f.write(f"## Top Performers\n\n")
            top = sorted(self.learnings, key=lambda x: x["metrics"]["total_pnl"], reverse=True)[:5]

            for l in top:
                f.write(f"### {l['test']}\n")
                f.write(f"- PnL: {l['metrics']['total_pnl']:,.0f}\n")
                f.write(f"- Sharpe: {l['metrics']['sharpe']:.2f}\n")
                f.write(f"- Hit: {l['metrics']['hit']:.2%}\n")
                f.write(f"- Insights:\n")
                for insight in l["insights"]:
                    f.write(f"  - {insight}\n")
                f.write(f"\n")

        print(f"📊 Relatório salvo em: {report_path}")

        # Salvar JSON também
        with open(self.session_dir / "learnings.json", "w") as f:
            json.dump(self.learnings, f, indent=2, default=str)


if __name__ == "__main__":
    system = RapidLearningSystem()
    system.run_rapid_learning()
