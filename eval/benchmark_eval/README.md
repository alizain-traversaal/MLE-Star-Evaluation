# Benchmark Evaluation Suite

This directory contains evaluation scripts and configurations for testing the MLE-STAR Multi Agent on various machine learning modeling benchmarks.

## Overview

The benchmark evaluation suite allows you to:
- Test the agent on multiple ML tasks from different benchmarks
- Evaluate performance across regression, classification, and other ML problem types
- Generate comprehensive evaluation reports
- Easily add new benchmark tasks

## Structure

- `benchmark_tasks.json`: Configuration file containing all benchmark tasks and their test queries
- `test_benchmark_eval.py`: Main evaluation script that runs tests on all configured benchmarks
- `test_config.json`: Evaluation criteria configuration
- `setup_mle_bench.py`: Helper script to download and set up MLE-Bench-Lite tasks

## Running Evaluations

### Run a Single Benchmark Task

```bash
pytest eval/benchmark_eval/test_benchmark_eval.py::test_benchmark_task -k "california-housing-prices"
```

### Run All Benchmark Tasks

```bash
pytest eval/benchmark_eval/test_benchmark_eval.py::test_all_benchmarks -v
```

### Run Individual Task Tests

```bash
pytest eval/benchmark_eval/test_benchmark_eval.py -v
```

## Adding New Benchmark Tasks

To add a new benchmark task:

1. **Prepare the task data:**
   - Create a directory under `machine_learning_engineering/tasks/` with the task name
   - Add `task_description.txt` with the task description
   - Add training and test data files (e.g., `train.csv`, `test.csv`)

2. **Add task configuration to `benchmark_tasks.json`:**
   ```json
   {
     "task_name": "your-task-name",
     "task_type": "Tabular Regression",  // or "Tabular Classification", etc.
     "metric": "root_mean_squared_error",
     "lower_is_better": true,
     "description": "Brief description of the task",
     "queries": [
       {
         "query": "describe the task that you have",
         "expected_tool_use": [],
         "expected_intermediate_agent_responses": [],
         "reference": "Expected response description"
       },
       {
         "query": "execute the task",
         "expected_tool_use": [],
         "expected_intermediate_agent_responses": [],
         "reference": "Expected execution description"
       }
     ]
   }
   ```

3. **Run the evaluation:**
   ```bash
   pytest eval/benchmark_eval/test_benchmark_eval.py -k "your-task-name"
   ```

## MLE-Bench-Lite Integration

The MLE-Bench-Lite benchmark contains 33 machine learning tasks from Kaggle competitions. To set up MLE-Bench-Lite tasks:

1. **Install MLE-Bench (if needed):**
   ```bash
   pip install mle-bench
   ```

2. **Use the setup script:**
   ```bash
   python eval/benchmark_eval/setup_mle_bench.py --tasks all
   # Or specify specific tasks:
   python eval/benchmark_eval/setup_mle_bench.py --tasks task1 task2 task3
   ```

3. **Add tasks to `benchmark_tasks.json`** following the format above.

## Evaluation Metrics

The evaluation uses the ADK `AgentEvaluator` which assesses:
- **Response Match Score**: How well the agent's responses match expected references
- **Tool Trajectory Score**: Quality of tool usage and execution flow

Results are saved to `benchmark_results.json` after running `test_all_benchmarks`.

## Configuration

You can adjust evaluation parameters in `test_config.json`:
- `tool_trajectory_avg_score`: Weight for tool usage evaluation (default: 0.6)
- `response_match_score`: Weight for response matching (default: 0.4)

Task-specific configurations (timeout, model, etc.) can be adjusted in the test functions or via environment variables.

## Notes

- Each benchmark task evaluation may take 15-30 minutes depending on task complexity
- Ensure sufficient compute resources and API quotas for running multiple benchmarks
- Results are saved incrementally, so you can resume evaluation if interrupted

