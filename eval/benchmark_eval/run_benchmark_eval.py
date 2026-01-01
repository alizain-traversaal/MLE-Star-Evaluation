"""Standalone script to run comprehensive benchmark evaluation.

This script provides a command-line interface for running benchmark evaluations
with various options and generating detailed reports.
"""

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

import dotenv
from google.adk.evaluation.agent_evaluator import AgentEvaluator

from machine_learning_engineering.shared_libraries import config


def load_benchmark_tasks(benchmark_file: Optional[pathlib.Path] = None) -> List[Dict[str, Any]]:
    """Load benchmark task configurations."""
    if benchmark_file is None:
        benchmark_file = pathlib.Path(__file__).parent / "benchmark_tasks.json"
    
    if not benchmark_file.exists():
        print(f"Error: Benchmark file not found: {benchmark_file}")
        sys.exit(1)
    
    with open(benchmark_file, "r") as f:
        return json.load(f)


def filter_tasks(tasks: List[Dict[str, Any]], task_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Filter tasks by name if specified."""
    if task_names is None:
        return tasks
    
    filtered = [task for task in tasks if task["task_name"] in task_names]
    if not filtered:
        print(f"Warning: No matching tasks found for: {task_names}")
    return filtered


async def run_evaluation(
    task_config: Dict[str, Any],
    num_runs: int = 1,
    timeout: int = 1800,
    output_dir: Optional[pathlib.Path] = None
) -> Dict[str, Any]:
    """Run evaluation on a single task.
    
    Args:
        task_config: Task configuration dictionary
        num_runs: Number of evaluation runs
        timeout: Execution timeout in seconds
        output_dir: Directory to save results
        
    Returns:
        Evaluation result dictionary
    """
    task_name = task_config["task_name"]
    print(f"\n{'='*60}")
    print(f"Evaluating task: {task_name}")
    print(f"  Type: {task_config['task_type']}")
    print(f"  Metric: {task_config['metric']}")
    print(f"{'='*60}")
    
    # Update config for this task
    config.CONFIG.task_name = task_config["task_name"]
    config.CONFIG.task_type = task_config["task_type"]
    config.CONFIG.lower = task_config["lower_is_better"]
    config.CONFIG.exec_timeout = timeout
    
    # Create test JSON file
    test_file = pathlib.Path(__file__).parent / f"{task_name}.test.json"
    with open(test_file, "w") as f:
        json.dump(task_config["queries"], f, indent=2)
    
    # Run evaluation
    start_time = time.time()
    try:
        await AgentEvaluator.evaluate(
            "machine_learning_engineering",
            str(test_file),
            num_runs=num_runs,
        )
        elapsed_time = time.time() - start_time
        status = "completed"
        error = None
    except Exception as e:
        elapsed_time = time.time() - start_time
        status = "failed"
        error = str(e)
        print(f"  ✗ Evaluation failed: {e}")
    
    result = {
        "task_name": task_name,
        "status": status,
        "elapsed_time": elapsed_time,
        "timestamp": datetime.now().isoformat(),
    }
    
    if error:
        result["error"] = error
    
    # Save individual result if output directory specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        result_file = output_dir / f"{task_name}_result.json"
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
    
    return result


async def run_all_evaluations(
    tasks: List[Dict[str, Any]],
    num_runs: int = 1,
    timeout: int = 1800,
    output_dir: Optional[pathlib.Path] = None,
    continue_on_error: bool = True
) -> List[Dict[str, Any]]:
    """Run evaluations on all specified tasks.
    
    Args:
        tasks: List of task configurations
        num_runs: Number of evaluation runs per task
        timeout: Execution timeout in seconds
        output_dir: Directory to save results
        continue_on_error: Whether to continue if a task fails
        
    Returns:
        List of evaluation results
    """
    results = []
    total_tasks = len(tasks)
    
    print(f"\nStarting evaluation of {total_tasks} task(s)...")
    print(f"Timeout per task: {timeout}s ({timeout/60:.1f} minutes)")
    print(f"Runs per task: {num_runs}\n")
    
    for idx, task_config in enumerate(tasks, 1):
        print(f"\n[{idx}/{total_tasks}] Processing {task_config['task_name']}...")
        
        try:
            result = await run_evaluation(
                task_config,
                num_runs=num_runs,
                timeout=timeout,
                output_dir=output_dir
            )
            results.append(result)
            
            if result["status"] == "completed":
                print(f"  ✓ Completed in {result['elapsed_time']:.1f}s")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
                if not continue_on_error:
                    print("Stopping evaluation due to error.")
                    break
        except KeyboardInterrupt:
            print("\n\nEvaluation interrupted by user.")
            break
        except Exception as e:
            print(f"  ✗ Unexpected error: {e}")
            results.append({
                "task_name": task_config["task_name"],
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            if not continue_on_error:
                break
    
    return results


def generate_report(results: List[Dict[str, Any]], output_file: pathlib.Path):
    """Generate a comprehensive evaluation report.
    
    Args:
        results: List of evaluation results
        output_file: Path to save the report
    """
    total = len(results)
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "failed")
    errors = sum(1 for r in results if r["status"] == "error")
    
    total_time = sum(r.get("elapsed_time", 0) for r in results)
    
    report = {
        "summary": {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "errors": errors,
            "success_rate": f"{(completed/total*100):.1f}%" if total > 0 else "0%",
            "total_time_seconds": total_time,
            "total_time_minutes": total_time / 60,
            "average_time_per_task": total_time / total if total > 0 else 0,
            "timestamp": datetime.now().isoformat(),
        },
        "results": results
    }
    
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks:        {total}")
    print(f"Completed:          {completed} ({(completed/total*100):.1f}%)" if total > 0 else "0")
    print(f"Failed:             {failed}")
    print(f"Errors:             {errors}")
    print(f"Total time:         {total_time/60:.1f} minutes")
    print(f"Average per task:   {total_time/total/60:.1f} minutes" if total > 0 else "N/A")
    print(f"\nReport saved to: {output_file}")
    print(f"{'='*60}\n")


async def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run comprehensive benchmark evaluation for MLE-STAR agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all benchmarks
  python run_benchmark_eval.py

  # Run specific tasks
  python run_benchmark_eval.py --tasks california-housing-prices

  # Run with custom timeout and multiple runs
  python run_benchmark_eval.py --timeout 3600 --runs 3

  # Run and save results to custom directory
  python run_benchmark_eval.py --output-dir ./eval_results
        """
    )
    
    parser.add_argument(
        "--tasks",
        nargs="+",
        help="Specific task names to evaluate (default: all tasks)"
    )
    parser.add_argument(
        "--benchmark-file",
        type=pathlib.Path,
        help="Path to benchmark tasks JSON file (default: benchmark_tasks.json)"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of evaluation runs per task (default: 1)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Execution timeout per task in seconds (default: 1800 = 30 minutes)"
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        help="Directory to save individual task results (default: ./benchmark_results)"
    )
    parser.add_argument(
        "--report-file",
        type=pathlib.Path,
        help="Path to save evaluation report (default: ./benchmark_results/report.json)"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop evaluation if a task fails"
    )
    
    args = parser.parse_args()
    
    # Load environment
    dotenv.load_dotenv()
    
    # Load benchmark tasks
    tasks = load_benchmark_tasks(args.benchmark_file)
    
    # Filter tasks if specified
    if args.tasks:
        tasks = filter_tasks(tasks, args.tasks)
        if not tasks:
            print("No matching tasks found. Exiting.")
            return 1
    
    if not tasks:
        print("No tasks to evaluate. Exiting.")
        return 1
    
    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = pathlib.Path(__file__).parent / "benchmark_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set report file
    if args.report_file:
        report_file = args.report_file
    else:
        report_file = output_dir / "report.json"
    
    # Run evaluations
    results = await run_all_evaluations(
        tasks=tasks,
        num_runs=args.runs,
        timeout=args.timeout,
        output_dir=output_dir,
        continue_on_error=not args.stop_on_error
    )
    
    # Generate report
    generate_report(results, report_file)
    
    return 0


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))

