"""Helper script to download and set up Kaggle datasets for evaluation.

This script downloads a Kaggle dataset using the Kaggle API and sets it up
in the format required by the MLE-STAR agent.
"""

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Optional

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    pass  # dotenv not required if env vars are set directly

TASKS_DIR = pathlib.Path(__file__).parent.parent.parent / "machine_learning_engineering" / "tasks"
BENCHMARK_CONFIG = pathlib.Path(__file__).parent / "benchmark_tasks.json"


def check_kaggle_installed() -> bool:
    """Check if kaggle package is installed."""
    try:
        import kaggle
        return True
    except ImportError:
        return False


def install_kaggle():
    """Install the kaggle package."""
    print("Installing kaggle package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
        print("✓ kaggle installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install kaggle: {e}")
        return False


def setup_kaggle_credentials():
    """Set up Kaggle API credentials from environment variables."""
    # Ensure .env is loaded
    try:
        import dotenv
        dotenv.load_dotenv()
    except ImportError:
        pass
    
    # Kaggle expects credentials in ~/.kaggle/kaggle.json
    kaggle_dir = pathlib.Path.home() / ".kaggle"
    kaggle_dir.mkdir(exist_ok=True)
    kaggle_json = kaggle_dir / "kaggle.json"
    
    # Try different credential formats
    username = None
    key = None
    
    # Option 1: Separate environment variables (KAGGLE_USERNAME + KAGGLE_KEY)
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        username = os.getenv("KAGGLE_USERNAME")
        key = os.getenv("KAGGLE_KEY")
    # Option 2: KAGGLE_USERNAME + KAGGLE_API_TOKEN (token as key)
    elif os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_API_TOKEN"):
        username = os.getenv("KAGGLE_USERNAME")
        kaggle_token = os.getenv("KAGGLE_API_TOKEN")
        # If token contains colon, it's username:key format, extract just the key
        if ":" in kaggle_token:
            _, key = kaggle_token.split(":", 1)
        else:
            key = kaggle_token
    # Option 3: Combined token (username:key format)
    elif os.getenv("KAGGLE_API_TOKEN"):
        kaggle_token = os.getenv("KAGGLE_API_TOKEN")
        if ":" in kaggle_token:
            username, key = kaggle_token.split(":", 1)
        else:
            print("Error: KAGGLE_API_TOKEN doesn't contain username:key format")
            print("Please set KAGGLE_USERNAME environment variable or use 'username:key' format")
            return False
    
    if not username or not key:
        print("Error: Kaggle credentials not found")
        print("Please set one of:")
        print("  - KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print("  - KAGGLE_API_TOKEN in format 'username:key'")
        return False
    
    credentials = {
        "username": username,
        "key": key
    }
    
    with open(kaggle_json, "w") as f:
        json.dump(credentials, f)
    
    # Set proper permissions (Kaggle requires 600 on Unix, but Windows doesn't support chmod the same way)
    try:
        os.chmod(kaggle_json, 0o600)
    except (AttributeError, NotImplementedError):
        # Windows doesn't support chmod, which is okay
        pass
    
    print("✓ Kaggle credentials configured")
    return True


def download_kaggle_dataset(
    dataset: str,
    task_name: str,
    output_dir: pathlib.Path
) -> bool:
    """Download a Kaggle dataset.
    
    Args:
        dataset: Kaggle dataset identifier (e.g., 'username/dataset-name' or competition name)
        task_name: Name to use for the task directory
        output_dir: Directory where task should be saved
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        task_dir = output_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Downloading Kaggle dataset: {dataset}...")
        print(f"Destination: {task_dir}")
        
        # Download dataset
        api.dataset_download_files(dataset, path=str(task_dir), unzip=True)
        
        # Move files from subdirectory if needed
        # Kaggle sometimes creates a subdirectory with the dataset name
        subdirs = [d for d in task_dir.iterdir() if d.is_dir()]
        if len(subdirs) == 1 and not any(task_dir.glob("*.csv")):
            # Move files from subdirectory to task_dir
            subdir = subdirs[0]
            for file in subdir.iterdir():
                shutil.move(str(file), str(task_dir / file.name))
            subdir.rmdir()
        
        print(f"✓ Dataset downloaded successfully")
        return True
    except Exception as e:
        print(f"✗ Failed to download dataset: {e}")
        return False


def create_task_description(
    task_name: str,
    task_dir: pathlib.Path,
    task_type: str = "Tabular Regression",
    metric: str = "root_mean_squared_error",
    description: Optional[str] = None
) -> bool:
    """Create a task_description.txt file.
    
    Args:
        task_name: Name of the task
        task_dir: Directory where task files are located
        task_type: Type of ML task
        metric: Evaluation metric
        description: Optional custom description
        
    Returns:
        True if successful
    """
    task_desc_file = task_dir / "task_description.txt"
    
    # Try to infer target column and features from data files
    target_column = "target"
    train_file = None
    
    # Look for train.csv or similar
    for pattern in ["train.csv", "train*.csv", "*.csv"]:
        matches = list(task_dir.glob(pattern))
        if matches:
            train_file = matches[0]
            break
    
    if train_file and train_file.exists():
        try:
            import pandas as pd
            df = pd.read_csv(train_file, nrows=5)  # Read just headers
            columns = df.columns.tolist()
            # Common target column names
            target_candidates = ["target", "label", "y", task_name.split("-")[-1]]
            for candidate in target_candidates:
                if candidate in columns:
                    target_column = candidate
                    break
        except Exception:
            pass
    
    # Create task description
    if description:
        desc_text = description
    else:
        desc_text = f"Predict the {target_column} for {task_name.replace('-', ' ')}."
    
    content = f"""# Task

{desc_text}

# Metric

{metric}

# Submission Format

```
{target_column}
value1
value2
etc.
```

# Dataset

"""
    
    # List data files
    csv_files = list(task_dir.glob("*.csv"))
    for csv_file in csv_files:
        filename = csv_file.name
        if "train" in filename.lower():
            content += f"{filename}\n"
            content += f"Contains features and {target_column} target variable.\n\n"
        elif "test" in filename.lower():
            content += f"{filename}\n"
            content += f"Contains features only (no target variable).\n"
            content += f"Same feature columns as training data.\n\n"
        else:
            content += f"{filename}\n"
            content += f"Data file for {task_name}.\n\n"
    
    with open(task_desc_file, "w") as f:
        f.write(content)
    
    print(f"✓ Created task_description.txt")
    return True


def create_task_config(
    task_name: str,
    task_type: str = "Tabular Regression",
    metric: str = "root_mean_squared_error",
    lower_is_better: bool = True,
    description: Optional[str] = None
) -> dict:
    """Create a benchmark task configuration entry.
    
    Args:
        task_name: Name of the task
        task_type: Type of ML task
        metric: Evaluation metric
        lower_is_better: Whether lower metric values are better
        description: Optional task description
        
    Returns:
        Task configuration dictionary
    """
    if not description:
        description = f"Kaggle dataset: {task_name}"
    
    return {
        "task_name": task_name,
        "task_type": task_type,
        "metric": metric,
        "lower_is_better": lower_is_better,
        "description": description,
        "queries": [
            {
                "query": "describe the task that you have",
                "expected_tool_use": [],
                "expected_intermediate_agent_responses": [],
                "reference": f"The task I have is the {task_name.replace('-', ' ').title()} Task. This task involves predicting target values based on various features. It's a {task_type.lower()} problem where the goal is to build a model that can accurately estimate the target using {metric} metric."
            },
            {
                "query": f"execute the {task_name} task",
                "expected_tool_use": [],
                "expected_intermediate_agent_responses": [],
                "reference": f"{task_type} to predict target using {metric} metric."
            }
        ]
    }


def update_benchmark_config(task_config: dict):
    """Update benchmark_tasks.json with new task configuration.
    
    Args:
        task_config: Task configuration dictionary
    """
    if BENCHMARK_CONFIG.exists():
        with open(BENCHMARK_CONFIG, "r") as f:
            existing_tasks = json.load(f)
    else:
        existing_tasks = []
    
    # Check if task already exists
    existing_names = {task["task_name"] for task in existing_tasks}
    
    if task_config["task_name"] not in existing_names:
        existing_tasks.append(task_config)
        with open(BENCHMARK_CONFIG, "w") as f:
            json.dump(existing_tasks, f, indent=2)
        print(f"✓ Added {task_config['task_name']} to benchmark configuration")
    else:
        # Update existing task
        for i, task in enumerate(existing_tasks):
            if task["task_name"] == task_config["task_name"]:
                existing_tasks[i] = task_config
                break
        with open(BENCHMARK_CONFIG, "w") as f:
            json.dump(existing_tasks, f, indent=2)
        print(f"⊘ Updated {task_config['task_name']} in benchmark configuration")
    
    print(f"✓ Updated {BENCHMARK_CONFIG}")


def create_test_json(task_config: dict):
    """Create test JSON file for the task.
    
    Args:
        task_config: Task configuration dictionary
    """
    test_file = pathlib.Path(__file__).parent / f"{task_config['task_name']}.test.json"
    with open(test_file, "w") as f:
        json.dump(task_config["queries"], f, indent=2)
    print(f"✓ Created {test_file.name}")


def main():
    """Main function to set up Kaggle dataset."""
    parser = argparse.ArgumentParser(
        description="Download and set up a Kaggle dataset for evaluation"
    )
    parser.add_argument(
        "dataset",
        help="Kaggle dataset identifier (e.g., 'username/dataset-name' or competition name)"
    )
    parser.add_argument(
        "--task-name",
        help="Name to use for the task directory (default: derived from dataset name)"
    )
    parser.add_argument(
        "--task-type",
        default="Tabular Regression",
        choices=["Tabular Regression", "Tabular Classification", "Time Series"],
        help="Type of ML task (default: Tabular Regression)"
    )
    parser.add_argument(
        "--metric",
        default="root_mean_squared_error",
        help="Evaluation metric (default: root_mean_squared_error)"
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        default=True,
        help="Whether lower metric values are better (default: True)"
    )
    parser.add_argument(
        "--description",
        help="Custom task description"
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip checking/installing kaggle package"
    )
    
    args = parser.parse_args()
    
    # Determine task name
    if args.task_name:
        task_name = args.task_name
    else:
        # Derive from dataset name
        task_name = args.dataset.split("/")[-1].lower().replace("_", "-")
    
    # Check/install kaggle
    if not args.skip_install:
        if not check_kaggle_installed():
            if not install_kaggle():
                print("Please install kaggle manually: pip install kaggle")
                return 1
    
    # Setup credentials
    if not setup_kaggle_credentials():
        return 1
    
    # Ensure tasks directory exists
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSetting up Kaggle dataset: {args.dataset}")
    print(f"Task name: {task_name}")
    print(f"Tasks directory: {TASKS_DIR}\n")
    
    # Download dataset
    if not download_kaggle_dataset(args.dataset, task_name, TASKS_DIR):
        return 1
    
    task_dir = TASKS_DIR / task_name
    
    # Create task description
    create_task_description(
        task_name,
        task_dir,
        task_type=args.task_type,
        metric=args.metric,
        description=args.description
    )
    
    # Create task config
    task_config = create_task_config(
        task_name,
        task_type=args.task_type,
        metric=args.metric,
        lower_is_better=args.lower_is_better,
        description=args.description
    )
    
    # Update benchmark config
    update_benchmark_config(task_config)
    
    # Create test JSON
    create_test_json(task_config)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Setup Summary:")
    print(f"  ✓ Dataset downloaded: {args.dataset}")
    print(f"  ✓ Task directory: {task_dir}")
    print(f"  ✓ Task configuration added to benchmark_tasks.json")
    print(f"  ✓ Test file created: {task_name}.test.json")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print("1. Review and update task_description.txt if needed:")
    print(f"   {task_dir / 'task_description.txt'}")
    print("2. Review benchmark configuration:")
    print(f"   {BENCHMARK_CONFIG}")
    print("3. Run evaluation:")
    print(f"   python eval/benchmark_eval/run_benchmark_eval.py --tasks {task_name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

