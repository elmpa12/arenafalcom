#!/usr/bin/env python3
"""
MAESTRO SESSION - Orquestrador Multi-AI
Claude 1 (Maestro) coordena Claude 2 (Estrategista) + GPT-5 (Crítico)

Objetivo: 500 micro-backtests em 5 segmentos de 100 cada
"""

import json
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List

class MaestroSession:
    """Claude 1 - Maestro & Orchestrator"""

    def __init__(self, session_id=None):
        self.session_id = session_id or datetime.now().strftime("%Y-%m-%d_%H%M")
        self.session_dir = Path(f"sessions/session_{self.session_id}")
        self.session_dir.mkdir(parents=True, exist_ok=True)

        # Session config
        self.config = {
            "tf": "5m",
            "total_tests": 500,
            "segments": 5,
            "tests_per_segment": 100,
            "parallel_batch": 16,  # Use 32 cores (16 * 2 cores each)

            # Micro-backtest config
            "window_days": 15,
            "train_days": 10,
            "val_days": 3,
            "step_days": 2,
            "episode_bars": 120,  # Mini-partidas

            # Targets
            "targets": {
                "hit": 0.52,
                "payoff": 1.25,
                "maxdd": -2000,
                "blunders_per_100": 2,
                "vol_bin_high_pct": 30.0
            }
        }

        # Save config
        with open(self.session_dir / "session_config.json", "w") as f:
            json.dump(self.config, f, indent=2)

        # Initialize logs
        self.logs_dir = self.session_dir / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        print(f"🎭 MAESTRO SESSION INITIALIZED")
        print(f"   Session ID: {self.session_id}")
        print(f"   Session Dir: {self.session_dir}")
        print(f"   Config: {self.config}")

    def create_segment_plan(self, segment_num: int, strategy_proposals: List[Dict]) -> Dict:
        """
        Maestro creates plan for segment based on Strategist proposals

        Args:
            segment_num: Segment number (1-5)
            strategy_proposals: Proposals from Claude 2 (Strategist)

        Returns:
            Segment plan with tests to execute
        """
        segment_dir = self.session_dir / f"segment_{segment_num}"
        segment_dir.mkdir(exist_ok=True)

        # Generate test configurations
        tests = []

        # For now, generate 100 micro-backtests with different periods
        # Each test = 15 days window, moving by 2 days step
        start_date = "2024-01-01"

        for i in range(self.config["tests_per_segment"]):
            # Calculate rolling window
            day_offset = i * self.config["step_days"]

            # Vary methods and parameters
            methods = ["macd_trend", "ema_crossover", "trend_breakout", "vwap_trend", "rsi_reversion"]
            method = methods[i % len(methods)]

            test_config = {
                "test_id": f"seg{segment_num}_test{i:03d}",
                "tf": self.config["tf"],
                "window_days": self.config["window_days"],
                "train_days": self.config["train_days"],
                "val_days": self.config["val_days"],
                "day_offset": day_offset,
                "method": method,
                "episode_bars": self.config["episode_bars"],

                # Parameters from strategy proposals
                "params": {
                    "timeout_mode": "bars",
                    "max_hold_bars": 480,
                    "atr_stop_mult": 1.5 + (i % 10) * 0.1,  # Vary 1.5-2.4
                    "hard_tp_usd": 200 + (i % 5) * 100,  # Vary 200-600
                    "use_atr_stop": i % 2 == 0,
                    "use_gates": i % 3 == 0
                }
            }

            tests.append(test_config)

        plan = {
            "segment": segment_num,
            "total_tests": len(tests),
            "tests": tests,
            "targets": self.config["targets"]
        }

        # Save plan
        with open(segment_dir / "segment_plan.json", "w") as f:
            json.dump(plan, f, indent=2)

        return plan

    def execute_segment(self, segment_num: int, plan: Dict) -> Dict:
        """
        Execute segment tests in parallel

        Returns:
            Segment results with metrics
        """
        print(f"\n{'='*80}")
        print(f"🚀 EXECUTING SEGMENT {segment_num}/{self.config['segments']}")
        print(f"{'='*80}")

        segment_dir = self.session_dir / f"segment_{segment_num}"

        # Create micro-backtest runner command for each test
        processes = []
        test_results = []

        for i, test in enumerate(plan["tests"]):
            # For now, simulate with rapid selector21 calls
            # In real implementation, would call micro_backtest_runner.py

            test_dir = segment_dir / test["test_id"]
            test_dir.mkdir(exist_ok=True)

            # Build selector21 command for this micro-test
            cmd = self._build_micro_test_command(test, test_dir)

            # Execute in batches of parallel_batch
            if len(processes) >= self.config["parallel_batch"]:
                # Wait for one to complete
                self._wait_for_batch(processes, test_results)

            # Start new process
            log_file = test_dir / "test.log"
            proc = subprocess.Popen(
                cmd,
                stdout=open(log_file, "w"),
                stderr=subprocess.STDOUT,
                cwd="/opt/botscalpv3"
            )

            processes.append({
                "proc": proc,
                "test_id": test["test_id"],
                "test_config": test,
                "start_time": time.time(),
                "test_dir": test_dir
            })

            print(f"[{i+1}/{len(plan['tests'])}] ▶️  {test['test_id']} ({test['method']})")

        # Wait for remaining processes
        while processes:
            self._wait_for_batch(processes, test_results)

        # Analyze results
        segment_results = self._analyze_segment_results(segment_num, test_results)

        # Save results
        with open(segment_dir / "segment_results.json", "w") as f:
            json.dump(segment_results, f, indent=2)

        return segment_results

    def _build_micro_test_command(self, test: Dict, test_dir: Path) -> List[str]:
        """Build selector21 command for micro-test"""

        # Calculate date range
        # For simplicity, use fixed range for now
        # Real implementation would calculate based on day_offset

        cmd = [
            "python3", "selector21.py",
            "--umcsv_root", "./data_monthly",
            "--symbol", "BTCUSDT",
            "--start", "2024-01-01",
            "--end", "2024-01-15",  # 15 days
            "--exec_rules", test["tf"],
            "--methods", test["method"],
            "--run_base",
            "--n_jobs", "2",
            "--out_root", str(test_dir),

            # Micro-backtest specific params
            "--timeout_mode", test["params"]["timeout_mode"],
            "--max_hold_bars", str(test["params"]["max_hold_bars"]),
        ]

        return cmd

    def _wait_for_batch(self, processes: List, results: List):
        """Wait for at least one process to complete"""
        while True:
            time.sleep(0.5)

            for p_info in processes:
                if p_info["proc"].poll() is not None:
                    # Process completed
                    elapsed = time.time() - p_info["start_time"]

                    # Read results
                    csv_path = p_info["test_dir"] / "leaderboard_base.csv"

                    result = {
                        "test_id": p_info["test_id"],
                        "test_config": p_info["test_config"],
                        "elapsed": elapsed,
                        "success": p_info["proc"].returncode == 0,
                        "metrics": self._read_test_metrics(csv_path) if csv_path.exists() else None
                    }

                    results.append(result)
                    processes.remove(p_info)

                    status = "✅" if result["success"] else "❌"
                    print(f"{status} {p_info['test_id']} ({elapsed:.1f}s)")

                    return

    def _read_test_metrics(self, csv_path: Path) -> Dict:
        """Extract metrics from leaderboard CSV"""
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            if len(df) > 0:
                row = df.iloc[0]
                return {
                    "hit": row["hit"],
                    "payoff": row["payoff"] if "payoff" in row else 0,
                    "total_pnl": row["total_pnl"],
                    "sharpe": row["sharpe"],
                    "maxdd": row["maxdd"],
                    "n_trades": row["n_trades"]
                }
        except Exception as e:
            print(f"   Warning: Could not read metrics from {csv_path}: {e}")

        return None

    def _analyze_segment_results(self, segment_num: int, test_results: List[Dict]) -> Dict:
        """Analyze segment results and generate summary"""

        successful = [r for r in test_results if r["success"] and r["metrics"]]

        if not successful:
            return {
                "segment": segment_num,
                "total_tests": len(test_results),
                "successful": 0,
                "metrics": {}
            }

        # Calculate averages
        avg_hit = sum(r["metrics"]["hit"] for r in successful) / len(successful)
        avg_payoff = sum(r["metrics"].get("payoff", 0) for r in successful) / len(successful)
        avg_pnl = sum(r["metrics"]["total_pnl"] for r in successful) / len(successful)
        avg_sharpe = sum(r["metrics"]["sharpe"] for r in successful) / len(successful)
        avg_maxdd = sum(r["metrics"]["maxdd"] for r in successful) / len(successful)

        # Find top performers
        profitable = [r for r in successful if r["metrics"]["total_pnl"] > 0]

        # Check targets
        targets_met = {
            "hit": avg_hit >= self.config["targets"]["hit"],
            "payoff": avg_payoff >= self.config["targets"]["payoff"],
            "maxdd": avg_maxdd >= self.config["targets"]["maxdd"]
        }

        return {
            "segment": segment_num,
            "total_tests": len(test_results),
            "successful": len(successful),
            "profitable": len(profitable),
            "profitable_pct": len(profitable) / len(successful) * 100 if successful else 0,

            "metrics": {
                "avg_hit": round(avg_hit, 4),
                "avg_payoff": round(avg_payoff, 4),
                "avg_pnl": round(avg_pnl, 2),
                "avg_sharpe": round(avg_sharpe, 4),
                "avg_maxdd": round(avg_maxdd, 2)
            },

            "targets_met": targets_met,
            "targets_met_pct": sum(targets_met.values()) / len(targets_met) * 100,

            "top_performers": sorted(
                profitable,
                key=lambda x: x["metrics"]["total_pnl"],
                reverse=True
            )[:10]
        }

    def run_session(self):
        """Run complete session (5 segments of 100 tests each)"""

        print(f"\n{'='*80}")
        print(f"🎭 MAESTRO SESSION START")
        print(f"{'='*80}")
        print(f"Session ID: {self.session_id}")
        print(f"Total Tests: {self.config['total_tests']}")
        print(f"Segments: {self.config['segments']}")
        print(f"Parallel: {self.config['parallel_batch']}")
        print(f"{'='*80}\n")

        session_results = []

        for segment_num in range(1, self.config['segments'] + 1):
            # Step 1: Get strategy proposals (simulate Claude 2 for now)
            proposals = self._get_strategy_proposals(segment_num)

            # Step 2: Create segment plan (Maestro)
            plan = self.create_segment_plan(segment_num, proposals)

            # Step 3: Execute segment
            segment_results = self.execute_segment(segment_num, plan)

            # Step 4: Print summary
            self._print_segment_summary(segment_results)

            session_results.append(segment_results)

        # Final session summary
        self._generate_session_summary(session_results)

    def _get_strategy_proposals(self, segment_num: int) -> List[Dict]:
        """Simulate Claude 2 (Strategist) proposals"""
        # For now, return baseline proposals
        # In real system, would query Claude 2 API

        return [
            {
                "proposal": "baseline",
                "methods": ["macd_trend", "ema_crossover", "trend_breakout"],
                "params": {"atr_stop_mult": 1.5, "hard_tp_usd": 200}
            },
            {
                "proposal": "aggressive",
                "methods": ["vwap_trend", "rsi_reversion"],
                "params": {"atr_stop_mult": 2.0, "hard_tp_usd": 400}
            },
            {
                "proposal": "conservative",
                "methods": ["ema_pullback", "orr_reversal"],
                "params": {"atr_stop_mult": 1.2, "hard_tp_usd": 150}
            }
        ]

    def _print_segment_summary(self, results: Dict):
        """Print segment summary"""
        print(f"\n{'='*80}")
        print(f"📊 SEGMENT {results['segment']} SUMMARY")
        print(f"{'='*80}")
        print(f"Tests: {results['successful']}/{results['total_tests']} successful")
        print(f"Profitable: {results['profitable']} ({results['profitable_pct']:.1f}%)")
        print(f"\nMetrics:")
        print(f"  Hit Rate: {results['metrics']['avg_hit']:.4f}")
        print(f"  Payoff: {results['metrics']['avg_payoff']:.4f}")
        print(f"  Avg PnL: {results['metrics']['avg_pnl']:,.2f}")
        print(f"  Avg Sharpe: {results['metrics']['avg_sharpe']:.4f}")
        print(f"  Avg MaxDD: {results['metrics']['avg_maxdd']:,.2f}")
        print(f"\nTargets Met: {results['targets_met_pct']:.0f}%")
        for target, met in results['targets_met'].items():
            status = "✅" if met else "❌"
            print(f"  {status} {target}: {met}")
        print(f"{'='*80}\n")

    def _generate_session_summary(self, all_results: List[Dict]):
        """Generate final session summary"""

        summary_path = self.session_dir / "session_summary.md"

        with open(summary_path, "w") as f:
            f.write(f"# Sessão {self.session_id}\n\n")
            f.write(f"**TF:** {self.config['tf']} | **Backtests:** {self.config['total_tests']}\n\n")

            # Overall metrics
            total_successful = sum(r['successful'] for r in all_results)
            total_profitable = sum(r['profitable'] for r in all_results)

            f.write(f"## Resultados Gerais\n\n")
            f.write(f"- Testes Completados: {total_successful}/{self.config['total_tests']}\n")
            f.write(f"- Testes Lucrativos: {total_profitable}\n")
            f.write(f"- Taxa de Sucesso: {total_profitable/total_successful*100:.1f}%\n\n")

            # Per segment
            f.write(f"## Por Segmento\n\n")
            for r in all_results:
                f.write(f"### Segmento {r['segment']}\n")
                f.write(f"- Hit: {r['metrics']['avg_hit']:.4f}\n")
                f.write(f"- Payoff: {r['metrics']['avg_payoff']:.4f}\n")
                f.write(f"- PnL Médio: {r['metrics']['avg_pnl']:,.2f}\n")
                f.write(f"- Sharpe: {r['metrics']['avg_sharpe']:.4f}\n")
                f.write(f"- Lucrativos: {r['profitable']}/{r['successful']} ({r['profitable_pct']:.1f}%)\n\n")

        print(f"\n✅ Session summary saved to: {summary_path}")


if __name__ == "__main__":
    maestro = MaestroSession()
    maestro.run_session()
