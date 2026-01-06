# Machine Learning Benchmark Evaluation Guide

This guide explains how to evaluate the MLE-STAR Multi Agent on machine learning modeling benchmarks.

## Quick Start

### 1. Basic Evaluation (Single Task)

Evaluate on the default California Housing Prices task:

```bash
pytest eval/benchmark_eval/test_benchmark_eval.py::test_benchmark_task -k "california-housing-prices" -v
```

### 2. Run All Configured Benchmarks

```bash
pytest eval/benchmark_eval/test_benchmark_eval.py::test_all_benchmarks -v
```

### 3. Using the Standalone Runner

```bash
# Run all benchmarks
python eval/benchmark_eval/run_benchmark_eval.py

# Run specific tasks
python eval/benchmark_eval/run_benchmark_eval.py --tasks california-housing-prices

# Custom timeout and runs
python eval/benchmark_eval/run_benchmark_eval.py --timeout 3600 --runs 3
```

## Adding Custom Benchmarks

### Step 1: Prepare Task Data

Create a directory structure:
```
machine_learning_engineering/tasks/your-task-name/
├── task_description.txt    # Task description and requirements
├── train.csv               # Training data
├── test.csv                # Test data (if applicable)
└── ...                     # Other data files
```

**task_description.txt format:**
```
# Task

Brief description of what to predict.

# Metric

Evaluation metric (e.g., root_mean_squared_error, accuracy, etc.)

# Submission Format

Expected output format.

# Dataset

Description of data files and format.
```

### Step 2: Add to Benchmark Configuration

Edit `eval/benchmark_eval/benchmark_tasks.json`:

```json
{
  "task_name": "your-task-name",
  "task_type": "Tabular Regression",  // or "Tabular Classification", etc.
  "metric": "root_mean_squared_error",
  "lower_is_better": true,  // true for RMSE, false for accuracy
  "description": "Brief description",
  "queries": [
    {
      "query": "describe the task that you have",
      "expected_tool_use": [],
      "expected_intermediate_agent_responses": [],
      "reference": "Expected response when describing the task"
    },
    {
      "query": "execute the your-task-name task",
      "expected_tool_use": [],
      "expected_intermediate_agent_responses": [],
      "reference": "Expected response when executing the task"
    }
  ]
}
```

### Step 3: Run Evaluation

```bash
pytest eval/benchmark_eval/test_benchmark_eval.py -k "your-task-name" -v
```

## Evaluation Metrics

The evaluation uses ADK's `AgentEvaluator` which assesses:

1. **Response Match Score** (40% weight by default):
   - Semantic similarity between agent responses and expected references
   - Uses embedding-based comparison

2. **Tool Trajectory Score** (60% weight by default):
   - Quality of tool usage
   - Correctness of execution flow
   - Proper use of sub-agents

### Customizing Evaluation Criteria

Edit `eval/benchmark_eval/test_config.json`:

```json
{
  "criteria": {
    "tool_trajectory_avg_score": 0.0,
    "response_match_score": 0.0
  }
}
```

## Configuration Options

### Task-Specific Configuration

Tasks can have different configurations. Modify in `test_benchmark_eval.py` or use environment variables:

- `exec_timeout`: Maximum execution time per task (default: 1800s = 30 minutes)
- `num_runs`: Number of evaluation runs per task (default: 1)
- `ROOT_AGENT_MODEL`: LLM model to use (default: gemini-2.5-flash)

### Agent Configuration

The agent's behavior can be configured via `config.CONFIG`:

- `num_solutions`: Number of different solutions to generate (default: 2)
- `num_model_candidates`: Number of model architectures to try (default: 2)
- `max_retry`: Maximum retry attempts (default: 10)
- `max_debug_round`: Maximum debugging iterations (default: 5)

## Understanding Results

### Individual Task Results

Each task evaluation generates:
- Test execution logs
- Agent response comparisons
- Tool usage analysis
- Performance metrics

### Summary Report

Running `test_all_benchmarks` or `run_benchmark_eval.py` generates:
- `benchmark_results.json`: Detailed results per task
- `report.json`: Summary with success rates and timing

### Report Structure

```json
{
  "summary": {
    "total_tasks": 5,
    "completed": 4,
    "failed": 1,
    "success_rate": "80.0%",
    "total_time_minutes": 120.5,
    "average_time_per_task": 24.1
  },
  "results": [
    {
      "task_name": "california-housing-prices",
      "status": "completed",
      "elapsed_time": 1800.5,
      "timestamp": "2025-01-XX..."
    },
    ...
  ]
}
```

## Best Practices

1. **Start Small**: Begin with a single task to verify setup
2. **Monitor Resources**: ML tasks can be compute-intensive
3. **Check Timeouts**: Adjust `exec_timeout` based on task complexity
4. **Review Logs**: Check workspace directories for detailed execution logs
5. **Iterate**: Refine task descriptions and queries based on results

## Troubleshooting

### Task Not Found

- Verify task directory exists in `machine_learning_engineering/tasks/`
- Check `task_description.txt` is present
- Ensure data files are in the correct location

### Evaluation Timeout

- Increase `exec_timeout` in test configuration
- Simplify task or reduce `num_solutions`/`num_model_candidates`

### Import Errors

- Ensure all dependencies are installed: `poetry install --with dev`
- Check environment variables are set correctly
- Verify ADK is properly configured

### Agent Execution Failures

- Check workspace logs in `machine_learning_engineering/workspace/{task_name}/`
- Review agent state in `final_state.json` files
- Verify data files are accessible and properly formatted

