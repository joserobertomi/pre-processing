import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def pareto_discretize(dataset: pd.DataFrame, threshold: float = 0.8, drop_others_dominant: bool = True, exclude_columns: List[str] = None) -> pd.DataFrame:
    """
    Discretize categorical columns using Pareto principle (80/20 rule).
    
    For each categorical column, keeps the top values that represent 'threshold' proportion
    of the data, and groups the remaining values into 'OTHERS_[column_name]' category.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Input dataset to be discretized
    threshold : float, default=0.8
        Proportion of data to keep as separate categories (0.8 = 80%)
    drop_others_dominant : bool, default=True
        If True, drops columns where OTHERS category has more count than individual top values
    exclude_columns : List[str], default=None
        List of column names to exclude from discretization (will be kept as-is)
    
    Returns:
    --------
    pd.DataFrame
        Discretized dataset with grouped categories (and potentially dropped columns)
    """
    
    # Create a copy to avoid modifying the original dataset
    discretized_dataset = dataset.copy()
    
    # Initialize exclude_columns as empty list if None
    if exclude_columns is None:
        exclude_columns = []
    
    # Track columns to drop
    columns_to_drop = []
    
    # Process each column
    for column in discretized_dataset.columns:
        # Skip excluded columns
        if column in exclude_columns:
            print(f"Column '{column}' excluded from discretization")
            continue
            
        # Check if column is categorical (object or category dtype)
        if discretized_dataset[column].dtype in ['object', 'category']:
            # Get value counts and calculate cumulative proportion
            value_counts = discretized_dataset[column].value_counts()
            total_count = len(discretized_dataset)
            
            # Calculate cumulative proportion
            cumulative_proportion = value_counts.cumsum() / total_count
            
            # Find values that exceed the threshold
            top_values = cumulative_proportion[cumulative_proportion <= threshold].index.tolist()
            
            # If no values meet the threshold, keep the most frequent one
            if not top_values and len(value_counts) > 0:
                top_values = [value_counts.index[0]]
            
            # Create a mapping: top values stay the same, others become "OTHERS_[column_name]"
            def map_values(value):
                if pd.isna(value):
                    return value
                return value if value in top_values else f"OTHERS_{column}"
            
            # Apply the mapping
            discretized_dataset[column] = discretized_dataset[column].apply(map_values)
            
            # Check if OTHERS category is dominant (has more count than individual top values)
            if drop_others_dominant:
                others_count = total_count - value_counts[top_values].sum()
                max_top_count = value_counts[top_values].max() if top_values else 0
                
                if others_count > max_top_count:
                    columns_to_drop.append(column)
                    print(f"Column '{column}' will be dropped: OTHERS count ({others_count}) > max top value count ({max_top_count})")
    
    # Drop columns where OTHERS is dominant
    if columns_to_drop:
        discretized_dataset = discretized_dataset.drop(columns=columns_to_drop)
        print(f"\nDropped {len(columns_to_drop)} columns where OTHERS category was dominant")
    
    return discretized_dataset

def get_discretization_summary(dataset: pd.DataFrame, threshold: float = 0.8) -> Dict:
    """
    Get a summary of how each column was discretized.
    
    Parameters:
    -----------
    dataset : pd.DataFrame
        Original dataset
    threshold : float, default=0.8
        Proportion threshold used for discretization
    
    Returns:
    --------
    Dict
        Summary of discretization for each column
    """
    
    summary = {}
    
    for column in dataset.columns:
        if dataset[column].dtype in ['object', 'category']:
            value_counts = dataset[column].value_counts()
            total_count = len(dataset)
            
            # Calculate cumulative proportion
            cumulative_proportion = value_counts.cumsum() / total_count
            
            # Find values that meet the threshold
            top_values = cumulative_proportion[cumulative_proportion <= threshold].index.tolist()
            
            # If no values meet the threshold, keep the most frequent one
            if not top_values and len(value_counts) > 0:
                top_values = [value_counts.index[0]]
            
            # Calculate counts for top values and others
            top_count = value_counts[top_values].sum()
            others_count = total_count - top_count
            
            summary[column] = {
                'original_unique_values': len(value_counts),
                'discretized_unique_values': len(top_values) + 1,  # +1 for OTHERS
                'top_values': top_values,
                'top_values_count': top_count,
                'others_count': others_count,
                'top_values_proportion': top_count / total_count,
                'others_proportion': others_count / total_count
            }
    
    return summary

# Example usage and testing
if __name__ == "__main__":
    # Example data similar to your aeronave_motor_quantidade
    example_data = {
        'aeronave_motor_quantidade': ['BIMOTOR'] * 2890 + ['MONOMOTOR'] * 1098 + ['SEM TRACAO'] * 218 + ['TRIMOTOR'] * 9,
        'example_bad_column': ['A'] * 100 + ['B'] * 50 + ['C'] * 30 + ['D'] * 20 + ['E'] * 15 + ['F'] * 10 + ['G'] * 5
    }
    
    df = pd.DataFrame(example_data)
    print("Original data:")
    print("\naeronave_motor_quantidade:")
    print(df['aeronave_motor_quantidade'].value_counts())
    print(f"Total: {len(df['aeronave_motor_quantidade'])}")
    
    print("\nexample_bad_column:")
    print(df['example_bad_column'].value_counts())
    print(f"Total: {len(df['example_bad_column'])}")
    print("\n" + "="*50 + "\n")
    
    # Apply Pareto discretization with automatic column dropping
    print("Applying Pareto discretization with automatic column dropping...")
    discretized_df = pareto_discretize(df, drop_others_dominant=True)
    
    print("\n" + "="*50 + "\n")
    
    # Example with excluded columns
    print("Example with excluded columns:")
    print("Excluding 'aeronave_motor_quantidade' from discretization...")
    discretized_df_excluded = pareto_discretize(df, drop_others_dominant=True, exclude_columns=['aeronave_motor_quantidade'])
    print(f"Final dataset shape with excluded columns: {discretized_df_excluded.shape}")
    print("Columns in result:", list(discretized_df_excluded.columns))
    print("\nAfter Pareto discretization:")
    
    if 'aeronave_motor_quantidade' in discretized_df.columns:
        print("\naeronave_motor_quantidade:")
        print(discretized_df['aeronave_motor_quantidade'].value_counts())
    
    if 'example_bad_column' in discretized_df.columns:
        print("\nexample_bad_column:")
        print(discretized_df['example_bad_column'].value_counts())
    
    print(f"\nFinal dataset shape: {discretized_df.shape}")
    print("\n" + "="*50 + "\n")
    
    # Get summary
    summary = get_discretization_summary(df)
    print("Discretization summary:")
    for col, info in summary.items():
        print(f"\n{col}:")
        print(f"  Original unique values: {info['original_unique_values']}")
        print(f"  Discretized unique values: {info['discretized_unique_values']}")
        print(f"  Top values: {info['top_values']}")
        print(f"  Top values proportion: {info['top_values_proportion']:.3f}")
        print(f"  Others proportion: {info['others_proportion']:.3f}")
        
        # Check if this column would be dropped
        others_count = info['others_count']
        max_top_count = max([info['top_values_count']] if info['top_values'] else [0])
        if others_count > max_top_count:
            print(f"  ⚠️  This column would be dropped (OTHERS count > max top value count)")
