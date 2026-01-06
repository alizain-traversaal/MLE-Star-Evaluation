"""Split test.csv into four cumulative evaluation windows."""

import pathlib
import pandas as pd
from datetime import datetime, timedelta

def split_test_files():
    """Split test.csv into test.csv (7 days), test2.csv (14 days), test3.csv (21 days), test4.csv (28 days)."""
    task_dir = pathlib.Path(__file__).parent.parent.parent / "machine_learning_engineering" / "tasks" / "jack-in-the-box"
    
    # Read the current test.csv
    test_file = task_dir / "test.csv"
    df = pd.read_csv(test_file)
    
    # Convert date column
    df['BUSINESS_DATE'] = pd.to_datetime(df['BUSINESS_DATE'])
    
    print(f"Original test.csv has {len(df)} days")
    print(f"Date range: {df['BUSINESS_DATE'].min()} to {df['BUSINESS_DATE'].max()}")
    
    # Create cumulative windows
    windows = [
        {'days': 7, 'filename': 'test.csv', 'description': 'First 7 days'},
        {'days': 14, 'filename': 'test2.csv', 'description': 'First 14 days'},
        {'days': 21, 'filename': 'test3.csv', 'description': 'First 21 days'},
        {'days': 28, 'filename': 'test4.csv', 'description': 'First 28 days'},
    ]
    
    for window in windows:
        # Take first N days
        window_df = df.head(window['days']).copy()
        window_df['BUSINESS_DATE'] = window_df['BUSINESS_DATE'].dt.strftime('%Y-%m-%d')
        
        # Save to file
        output_file = task_dir / window['filename']
        window_df.to_csv(output_file, index=False)
        
        print(f"\nCreated {window['filename']}: {window['description']}")
        print(f"  Date range: {window_df['BUSINESS_DATE'].min()} to {window_df['BUSINESS_DATE'].max()}")
        print(f"  Number of days: {len(window_df)}")
    
    return True

if __name__ == "__main__":
    split_test_files()

