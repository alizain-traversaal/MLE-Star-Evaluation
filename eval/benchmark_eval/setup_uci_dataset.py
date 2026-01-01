"""Helper script to download and set up UCI datasets for evaluation."""

import argparse
import json
import pathlib
import urllib.request
import zipfile
import sys
import pandas as pd
import os
import shutil

TASKS_DIR = pathlib.Path(__file__).parent.parent.parent / "machine_learning_engineering" / "tasks"
BENCHMARK_CONFIG = pathlib.Path(__file__).parent / "benchmark_tasks.json"

# UCI Binary Classification Datasets
UCI_DATASETS = {
    "breast-cancer-wisconsin": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data",
        "description": "Predict breast cancer (malignant or benign) based on cell characteristics",
        "target": "Class",
        "task_type": "Tabular Classification",
        "metric": "roc_auc_score",
        "lower_is_better": False,
    },
    "adult": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "test_url": "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        "description": "Predict if income exceeds $50K/year based on census data",
        "target": "income",
        "task_type": "Tabular Classification",
        "metric": "roc_auc_score",
        "lower_is_better": False,
    },
    "bank-marketing": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip",
        "description": "Predict if client will subscribe to a term deposit",
        "target": "y",
        "task_type": "Tabular Classification",
        "metric": "roc_auc_score",
        "lower_is_better": False,
    },
    "iris": {
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "description": "Classify iris flowers into three species (setosa, versicolor, virginica) based on sepal and petal measurements",
        "target": "class",
        "task_type": "Tabular Classification",
        "metric": "accuracy",
        "lower_is_better": False,
    }
}


def download_file(url: str, output_path: pathlib.Path) -> bool:
    """Download a file from URL."""
    try:
        print(f"Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"[OK] Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to download: {e}")
        return False


def setup_breast_cancer_wisconsin(task_dir: pathlib.Path) -> bool:
    """Set up Breast Cancer Wisconsin dataset."""
    dataset_info = UCI_DATASETS["breast-cancer-wisconsin"]
    url = dataset_info["url"]
    
    # Download data file
    data_file = task_dir / "breast-cancer-wisconsin.data"
    if not download_file(url, data_file):
        return False
    
    # Read and process the data
    try:
        # Column names from UCI
        columns = ["ID", "Clump_Thickness", "Uniformity_Cell_Size", "Uniformity_Cell_Shape",
                  "Marginal_Adhesion", "Single_Epithelial_Size", "Bare_Nuclei",
                  "Bland_Chromatin", "Normal_Nucleoli", "Mitoses", "Class"]
        
        df = pd.read_csv(data_file, names=columns, na_values="?")
        
        # Handle missing values
        df = df.dropna()
        
        # Convert Class to binary (2=benign, 4=malignant -> 0=benign, 1=malignant)
        df["Class"] = (df["Class"] / 2 - 1).astype(int)
        
        # Remove ID column
        df = df.drop("ID", axis=1)
        
        # Split into train and test (80/20)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # Save train and test
        train_df.to_csv(task_dir / "train.csv", index=False)
        test_df.drop("Class", axis=1).to_csv(task_dir / "test.csv", index=False)
        test_df[["Class"]].to_csv(task_dir / "test_targets.csv", index=False)
        
        print("[OK] Processed and split dataset")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process dataset: {e}")
        return False


def setup_iris(task_dir: pathlib.Path) -> bool:
    """Set up Iris dataset."""
    dataset_info = UCI_DATASETS["iris"]
    url = dataset_info["url"]
    
    # Download data file
    data_file = task_dir / "iris.data"
    if not download_file(url, data_file):
        return False
    
    # Read and process the data
    try:
        # Column names from UCI
        columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
        
        df = pd.read_csv(data_file, names=columns)
        
        # Convert class to numeric for binary classification (setosa=0, others=1)
        # Or keep as multi-class - let's keep it as multi-class but use accuracy
        df["class"] = df["class"].astype("category").cat.codes
        
        # Split into train and test (80/20)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # Save train and test
        train_df.to_csv(task_dir / "train.csv", index=False)
        test_df.drop("class", axis=1).to_csv(task_dir / "test.csv", index=False)
        test_df[["class"]].to_csv(task_dir / "test_targets.csv", index=False)
        
        print("[OK] Processed and split dataset")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process dataset: {e}")
        return False


def setup_bank_marketing(task_dir: pathlib.Path) -> bool:
    """Set up Bank Marketing dataset."""
    dataset_info = UCI_DATASETS["bank-marketing"]
    url = dataset_info["url"]
    
    # Download zip file
    zip_file = task_dir / "bank.zip"
    if not download_file(url, zip_file):
        return False
    
    # Extract zip
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(task_dir)
        
        # Find the CSV file
        csv_files = list(task_dir.glob("*.csv"))
        if not csv_files:
            print("[ERROR] No CSV file found in zip")
            return False
        
        data_file = csv_files[0]
        
        # Read and process
        df = pd.read_csv(data_file, sep=";")
        
        # Convert target to binary (yes/no -> 1/0)
        df["y"] = (df["y"] == "yes").astype(int)
        
        # Split into train and test (80/20)
        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        
        # Save train and test
        train_df.to_csv(task_dir / "train.csv", index=False)
        test_df.drop("y", axis=1).to_csv(task_dir / "test.csv", index=False)
        test_df[["y"]].to_csv(task_dir / "test_targets.csv", index=False)
        
        # Clean up
        zip_file.unlink()
        for file in task_dir.glob("*.csv"):
            if file.name not in ["train.csv", "test.csv", "test_targets.csv"]:
                file.unlink()
        
        print("[OK] Processed and split dataset")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to process dataset: {e}")
        return False


def create_task_description(task_name: str, task_dir: pathlib.Path, dataset_info: dict) -> bool:
    """Create task_description.txt file."""
    task_desc_file = task_dir / "task_description.txt"
    
    content = f"""# Task

{dataset_info["description"]}

# Metric

{dataset_info["metric"]}

# Submission Format

```
{dataset_info["target"]}
0
1
0
etc.
```

# Dataset

train.csv
Contains features and {dataset_info["target"]} target variable.

test.csv
Contains features only (no target variable).
Same feature columns as train.csv.
"""
    
    with open(task_desc_file, "w") as f:
        f.write(content)
    
    print("[OK] Created task_description.txt")
    return True


def create_task_config(task_name: str, dataset_info: dict) -> dict:
    """Create a benchmark task configuration entry."""
    return {
        "task_name": task_name,
        "task_type": dataset_info["task_type"],
        "metric": dataset_info["metric"],
        "lower_is_better": dataset_info["lower_is_better"],
        "description": dataset_info["description"],
        "queries": [
            {
                "query": "describe the task that you have",
                "expected_tool_use": [],
                "expected_intermediate_agent_responses": [],
                "reference": f"The task I have is the {task_name.replace('-', ' ').title()} Task. {dataset_info['description']}. It's a binary classification problem where the goal is to build a model that can accurately predict the target class using {dataset_info['metric']} metric."
            },
            {
                "query": f"execute the {task_name} task",
                "expected_tool_use": [],
                "expected_intermediate_agent_responses": [],
                "reference": f"{dataset_info['task_type']} to predict `{dataset_info['target']}` using `{dataset_info['metric']}` metric."
            }
        ]
    }


def update_benchmark_config(task_config: dict):
    """Update benchmark_tasks.json with new task configuration."""
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
        print(f"[OK] Added {task_config['task_name']} to benchmark configuration")
    else:
        # Update existing task
        for i, task in enumerate(existing_tasks):
            if task["task_name"] == task_config["task_name"]:
                existing_tasks[i] = task_config
                break
        with open(BENCHMARK_CONFIG, "w") as f:
            json.dump(existing_tasks, f, indent=2)
        print(f"[UPDATED] Updated {task_config['task_name']} in benchmark configuration")


def create_test_json(task_config: dict):
    """Create test JSON file for the task."""
    test_file = pathlib.Path(__file__).parent / f"{task_config['task_name']}.test.json"
    with open(test_file, "w") as f:
        json.dump(task_config["queries"], f, indent=2)
    print(f"[OK] Created {test_file.name}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Download and set up a UCI binary classification dataset"
    )
    parser.add_argument(
        "dataset",
        choices=list(UCI_DATASETS.keys()),
        help="UCI dataset to download"
    )
    
    args = parser.parse_args()
    
    dataset_info = UCI_DATASETS[args.dataset]
    task_name = args.dataset
    
    TASKS_DIR.mkdir(parents=True, exist_ok=True)
    task_dir = TASKS_DIR / task_name
    task_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSetting up UCI dataset: {task_name}")
    print(f"Description: {dataset_info['description']}")
    print(f"Tasks directory: {TASKS_DIR}\n")
    
    # Download and process dataset
    if task_name == "breast-cancer-wisconsin":
        success = setup_breast_cancer_wisconsin(task_dir)
    elif task_name == "bank-marketing":
        success = setup_bank_marketing(task_dir)
    elif task_name == "iris":
        success = setup_iris(task_dir)
    else:
        print(f"[ERROR] Dataset {task_name} setup not implemented yet")
        return 1
    
    if not success:
        return 1
    
    # Create task description
    create_task_description(task_name, task_dir, dataset_info)
    
    # Create task config
    task_config = create_task_config(task_name, dataset_info)
    
    # Update benchmark config
    update_benchmark_config(task_config)
    
    # Create test JSON
    create_test_json(task_config)
    
    print(f"\n{'='*60}")
    print(f"Setup Summary:")
    print(f"  [OK] Dataset downloaded and processed")
    print(f"  [OK] Task directory: {task_dir}")
    print(f"  [OK] Task configuration added to benchmark_tasks.json")
    print(f"  [OK] Test file created: {task_name}.test.json")
    print(f"{'='*60}\n")
    
    print("Next steps:")
    print(f"1. Review task_description.txt: {task_dir / 'task_description.txt'}")
    print(f"2. Run evaluation:")
    print(f"   python eval/benchmark_eval/run_benchmark_eval.py --tasks {task_name}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

