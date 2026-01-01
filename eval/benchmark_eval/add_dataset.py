"""Helper script to add a new dataset for evaluation.

This script helps set up a new ML task dataset by downloading from common sources.
"""

import argparse
import pathlib
import sys
import urllib.request
import io
import zipfile

try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    if PANDAS_AVAILABLE:
        print("Warning: scikit-learn not available. Some datasets may not be accessible.")


def download_wine_quality_dataset():
    """Download Wine Quality dataset from UCI."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    print(f"Downloading Wine Quality dataset from {url}...")
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(data), sep=';')
        return df
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None


def create_wine_quality_task(output_dir: pathlib.Path):
    """Create a Wine Quality regression task."""
    print("Creating Wine Quality regression task...")
    
    if not PANDAS_AVAILABLE:
        print("Error: pandas required for this dataset. Please install: pip install pandas")
        return None
    
    # Download or use sklearn dataset
    df = download_wine_quality_dataset()
    
    if df is None:
        # Fallback: try sklearn
        try:
            from sklearn.datasets import load_wine
            wine = load_wine()
            df = pd.DataFrame(wine.data, columns=wine.feature_names)
            df['quality'] = wine.target
        except:
            print("Error: Could not load wine dataset. Please ensure pandas and scikit-learn are installed.")
            return None
    
    # Separate features and target
    if 'quality' in df.columns:
        X = df.drop('quality', axis=1)
        y = df['quality']
    else:
        # If no quality column, use last column as target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        y.name = 'quality'
    
    # Split into train and test
    if SKLEARN_AVAILABLE:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # Manual split if sklearn not available
        split_idx = int(len(df) * 0.8)
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
    
    # Create task directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine features and target for training
    train_df = X_train.copy()
    train_df['quality'] = y_train
    train_df.to_csv(output_dir / 'train.csv', index=False)
    
    # Test set without target
    X_test.to_csv(output_dir / 'test.csv', index=False)
    # Save test targets separately for reference (not used by agent)
    y_test.to_csv(output_dir / 'test_targets.csv', index=False)
    
    # Create task description
    feature_names = ', '.join(X.columns.tolist()[:5]) + ', ...' if len(X.columns) > 5 else ', '.join(X.columns.tolist())
    task_desc = f"""# Task

Predict the wine quality.

# Metric

root_mean_squared_error

# Submission Format
```
quality
5.0
6.0
7.0
etc.
```

# Dataset

train.csv
Contains wine features and quality target variable.
Features include: {feature_names}

test.csv
Contains wine features only (no target variable).
Same feature columns as train.csv.
"""
    
    with open(output_dir / 'task_description.txt', 'w') as f:
        f.write(task_desc)
    
    print(f"[OK] Wine Quality task created in {output_dir}")
    return "wine-quality"


def create_boston_housing_task(output_dir: pathlib.Path):
    """Create a Boston Housing regression task."""
    print("Creating Boston Housing regression task...")
    
    if not SKLEARN_AVAILABLE:
        print("Error: pandas and sklearn required. Please install: pip install pandas scikit-learn")
        return None
    
    # Load Boston housing dataset (may not be available in newer sklearn)
    try:
        from sklearn.datasets import fetch_california_housing
        # Use California Housing as alternative
        print("Using California Housing as Boston Housing alternative...")
        california = fetch_california_housing()
        X = pd.DataFrame(california.data, columns=california.feature_names)
        y = pd.Series(california.target, name='medv')
        task_name_suffix = "variant"
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create task directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Combine features and target for training
    train_df = X_train.copy()
    train_df['medv'] = y_train
    train_df.to_csv(output_dir / 'train.csv', index=False)
    
    # Test set without target
    X_test.to_csv(output_dir / 'test.csv', index=False)
    y_test.to_csv(output_dir / 'test_targets.csv', index=False)
    
    # Create task description
    task_desc = """# Task

Predict the median value of owner-occupied homes (medv).

# Metric

root_mean_squared_error

# Submission Format
```
medv
2.344
1.876
3.421
etc.
```

# Dataset

train.csv
Contains housing features and medv target variable.

test.csv
Contains housing features only (no target variable).
"""
    
    with open(output_dir / 'task_description.txt', 'w') as f:
        f.write(task_desc)
    
    print(f"[OK] Housing task created in {output_dir}")
    return "boston-housing"


def create_california_housing_variant(output_dir: pathlib.Path):
    """Create a variant of California Housing with different split."""
    print("Creating California Housing variant task...")
    
    if not SKLEARN_AVAILABLE:
        print("Error: pandas and sklearn required. Please install: pip install pandas scikit-learn")
        return None
    
    try:
        from sklearn.datasets import fetch_california_housing
        california = fetch_california_housing()
        X = pd.DataFrame(california.data, columns=california.feature_names)
        y = pd.Series(california.target, name='MedHouseVal')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None
    
    # Use different random state for different split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123  # Different from default
    )
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = X_train.copy()
    train_df['MedHouseVal'] = y_train
    train_df.to_csv(output_dir / 'train.csv', index=False)
    
    X_test.to_csv(output_dir / 'test.csv', index=False)
    y_test.to_csv(output_dir / 'test_targets.csv', index=False)
    
    task_desc = """# Task

Predict the median house value (MedHouseVal).

# Metric

root_mean_squared_error

# Submission Format
```
MedHouseVal
2.344
1.876
3.421
etc.
```

# Dataset

train.csv
Contains housing features and MedHouseVal target variable.
Features include: MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude

test.csv
Contains housing features only (no target variable).
"""
    
    with open(output_dir / 'task_description.txt', 'w') as f:
        f.write(task_desc)
    
    print(f"[OK] California Housing variant created in {output_dir}")
    return "california-housing-variant"


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Add a new dataset for ML evaluation"
    )
    parser.add_argument(
        "--dataset",
        choices=["wine-quality", "boston-housing", "california-variant"],
        default="wine-quality",
        help="Dataset to create"
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=None,
        help="Output directory (default: machine_learning_engineering/tasks/{dataset_name})"
    )
    parser.add_argument(
        "--update-benchmark",
        action="store_true",
        help="Update benchmark_tasks.json with the new task"
    )
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        tasks_dir = pathlib.Path(__file__).parent.parent.parent / "machine_learning_engineering" / "tasks"
        args.output_dir = tasks_dir / args.dataset
    
    # Create dataset
    if args.dataset == "wine-quality":
        task_name = create_wine_quality_task(args.output_dir)
    elif args.dataset == "boston-housing":
        task_name = create_boston_housing_task(args.output_dir)
    elif args.dataset == "california-variant":
        task_name = create_california_housing_variant(args.output_dir)
    
    # Update benchmark config if requested
    if args.update_benchmark:
        import json
        benchmark_file = pathlib.Path(__file__).parent / "benchmark_tasks.json"
        
        if benchmark_file.exists():
            with open(benchmark_file, 'r') as f:
                tasks = json.load(f)
        else:
            tasks = []
        
        # Check if task already exists
        if not any(t.get("task_name") == task_name for t in tasks):
            task_config = {
                "task_name": task_name,
                "task_type": "Tabular Regression",
                "metric": "root_mean_squared_error",
                "lower_is_better": True,
                "description": f"Regression task: {task_name}",
                "queries": [
                    {
                        "query": "describe the task that you have",
                        "expected_tool_use": [],
                        "expected_intermediate_agent_responses": [],
                        "reference": f"Task description for {task_name}"
                    },
                    {
                        "query": f"execute the {task_name} task",
                        "expected_tool_use": [],
                        "expected_intermediate_agent_responses": [],
                        "reference": f"Tabular regression to predict target using root_mean_squared_error metric."
                    }
                ]
            }
            tasks.append(task_config)
            
            with open(benchmark_file, 'w') as f:
                json.dump(tasks, f, indent=2)
            
            print(f"[OK] Updated {benchmark_file} with {task_name}")
        else:
            print(f"[SKIP] {task_name} already in benchmark configuration")
    
    if task_name:
        print(f"\n[OK] Dataset setup complete!")
        print(f"  Task directory: {args.output_dir}")
        print(f"  Task name: {task_name}")
        print(f"\nNext steps:")
        print(f"  1. Review task_description.txt in {args.output_dir}")
        print(f"  2. Run evaluation: pytest eval/benchmark_eval/test_benchmark_eval.py -k '{task_name}' -v")
    else:
        print("\n[ERROR] Failed to create dataset. Please check error messages above.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

