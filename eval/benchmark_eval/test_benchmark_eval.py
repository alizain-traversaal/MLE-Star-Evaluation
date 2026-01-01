"""Comprehensive benchmark evaluation for Machine Learning Engineering Agent.

This evaluation suite tests the MLE-STAR agent on multiple machine learning
benchmark tasks, including tasks from MLE-Bench-Lite and other standard ML datasets.
"""

import json
import pathlib
from typing import Dict, List, Any

import dotenv
import pytest
from google.adk.evaluation.agent_evaluator import AgentEvaluator

from machine_learning_engineering.shared_libraries import config


pytest_plugins = ("pytest_asyncio",)


@pytest.fixture(scope="session", autouse=True)
def load_env():
    """Load environment variables from .env file."""
    dotenv.load_dotenv()


def load_benchmark_tasks() -> List[Dict[str, Any]]:
    """Load benchmark task configurations."""
    benchmark_file = pathlib.Path(__file__).parent / "benchmark_tasks.json"
    with open(benchmark_file, "r") as f:
        return json.load(f)


@pytest.mark.asyncio
@pytest.mark.parametrize("task_config", load_benchmark_tasks())
async def test_benchmark_task(monkeypatch, task_config: Dict[str, Any]):
    """Test the agent on a specific benchmark task.
    
    Args:
        monkeypatch: pytest fixture for patching
        task_config: Configuration dictionary for the benchmark task
    """
    # Set task-specific configuration
    monkeypatch.setattr(config.CONFIG, "task_name", task_config["task_name"])
    monkeypatch.setattr(config.CONFIG, "task_type", task_config["task_type"])
    monkeypatch.setattr(config.CONFIG, "lower", task_config["lower_is_better"])
    monkeypatch.setattr(config.CONFIG, "exec_timeout", 1800)  # 30 minutes for full task execution
    
    # Create test JSON file for this task
    test_file = pathlib.Path(__file__).parent / f"{task_config['task_name']}.test.json"
    with open(test_file, "w") as f:
        json.dump(task_config["queries"], f, indent=2)
    
    # Run evaluation
    await AgentEvaluator.evaluate(
        "machine_learning_engineering",
        str(test_file),
        num_runs=1,
    )


@pytest.mark.asyncio
async def test_all_benchmarks(monkeypatch):
    """Run evaluation on all configured benchmark tasks sequentially.
    
    This test runs all benchmark tasks and generates a comprehensive report.
    """
    benchmark_tasks = load_benchmark_tasks()
    results = []
    
    for task_config in benchmark_tasks:
        try:
            # Set task-specific configuration
            monkeypatch.setattr(config.CONFIG, "task_name", task_config["task_name"])
            monkeypatch.setattr(config.CONFIG, "task_type", task_config["task_type"])
            monkeypatch.setattr(config.CONFIG, "lower", task_config["lower_is_better"])
            monkeypatch.setattr(config.CONFIG, "exec_timeout", 1800)
            
            # Create test JSON file for this task
            test_file = pathlib.Path(__file__).parent / f"{task_config['task_name']}.test.json"
            with open(test_file, "w") as f:
                json.dump(task_config["queries"], f, indent=2)
            
            # Run evaluation
            await AgentEvaluator.evaluate(
                "machine_learning_engineering",
                str(test_file),
                num_runs=1,
            )
            
            results.append({
                "task_name": task_config["task_name"],
                "status": "completed"
            })
        except Exception as e:
            results.append({
                "task_name": task_config["task_name"],
                "status": "failed",
                "error": str(e)
            })
    
    # Save results summary
    results_file = pathlib.Path(__file__).parent / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nBenchmark evaluation complete. Results saved to {results_file}")

