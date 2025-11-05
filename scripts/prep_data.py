"""
Data Preparation Script for Predictive Maintenance Pipeline

This script loads raw data files based on a YAML configuration, performs basic
cleaning and data type validation, and saves the cleaned data to parquet format.

Usage:
    python scripts/prep_data.py --config configs/ims.yaml
    python scripts/prep_data.py --config configs/cwru.yaml
    python scripts/prep_data.py --config configs/ai4i.yaml
"""

import argparse
import pathlib
import re
from datetime import datetime
from typing import Dict, List, Optional, Union

import pandas as pd
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from tqdm import tqdm

console = Console()


def load_config(config_path: Union[str, pathlib.Path]) -> Dict:
    """Load and parse YAML configuration file."""
    config_path = pathlib.Path(config_path)
    
    # If config_path doesn't exist, try common variations
    if not config_path.exists():
        # Try with 'configs' directory (common typo: config vs configs)
        if 'config' in str(config_path) and 'configs' not in str(config_path):
            alt_path = pathlib.Path(str(config_path).replace('config/', 'configs/').replace('config\\', 'configs\\'))
            if alt_path.exists():
                console.print(f"[yellow]Note: Using {alt_path} instead of {config_path}[/yellow]")
                config_path = alt_path
            else:
                raise FileNotFoundError(
                    f"Config file not found: {config_path}\n"
                    f"  (Also tried: {alt_path})\n"
                    f"  Make sure the path is correct (use 'configs/' not 'config/')"
                )
        else:
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"  Make sure the path is correct"
            )
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def extract_timestamp_from_filename(filename: str, timestamp_format: str) -> Optional[datetime]:
    """
    Extract timestamp from filename based on format.
    
    For IMS: filename is the timestamp itself (e.g., '2003.10.22.12.06.24')
    """
    try:
        # Remove file extension if present
        base_name = pathlib.Path(filename).stem
        timestamp = datetime.strptime(base_name, timestamp_format)
        return timestamp
    except (ValueError, AttributeError):
        return None


def load_ims_data(config: Dict) -> pd.DataFrame:
    """
    Load IMS bearing data from timestamped files.
    
    Each file contains tab-separated vibration data with 8 channels.
    Files are named with timestamps showing the progression from healthy to failure.
    """
    raw_input_dir = pathlib.Path(config['paths']['raw_input_dir'])
    max_files = config['paths'].get('max_files')
    
    if not raw_input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {raw_input_dir}")
    
    # Get file format settings
    file_format = config.get('file_format', {})
    delimiter = file_format.get('delimiter', '\t')
    header = file_format.get('header', False)
    columns_in_order = file_format.get('columns_in_order', [])
    extract_timestamp = file_format.get('extract_timestamp_from_filename', False)
    timestamp_format = file_format.get('timestamp_format', '%Y.%m.%d.%H.%M.%S')
    
    # Get column mapping from schema
    column_map = config.get('schema', {}).get('column_map', {})
    
    # Get all files and sort them (filenames are timestamps, so sorting gives chronological order)
    all_files = sorted([f for f in raw_input_dir.iterdir() if f.is_file() and not f.name.startswith('.')])
    
    if max_files:
        all_files = all_files[:max_files]
    
    console.print(f"[green]Loading IMS data from {len(all_files)} files...[/green]")
    
    dataframes = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing files...", total=len(all_files))
        
        for file_path in all_files:
            try:
                # Load data
                df = pd.read_csv(
                    file_path,
                    delimiter=delimiter,
                    header=None if not header else 0,
                    names=columns_in_order if columns_in_order else None,
                    dtype=float,
                    on_bad_lines='skip',  # Skip malformed lines
                    engine='python'  # Use Python engine for better error handling
                )
                
                # Apply column mapping if provided
                if column_map and not columns_in_order:
                    # Map numeric column indices to names
                    df.columns = [column_map.get(str(i), f'col_{i}') for i in range(len(df.columns))]
                elif columns_in_order:
                    df.columns = columns_in_order
                
                # Extract timestamp from filename if configured
                if extract_timestamp:
                    timestamp = extract_timestamp_from_filename(file_path.name, timestamp_format)
                    if timestamp:
                        df['timestamp'] = timestamp
                        df['file_index'] = len(dataframes)  # Sequential file index
                
                # Add file identifier
                df['source_file'] = file_path.name
                
                # Ensure all numeric columns are actually numeric
                numeric_cols = config.get('schema', {}).get('numeric_features', [])
                if numeric_cols:
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if df.empty:
                    console.print(f"[yellow]Warning: File {file_path.name} is empty after loading[/yellow]")
                    continue
                
                dataframes.append(df)
                progress.update(task, advance=1)
                
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load {file_path.name}: {e}[/yellow]")
                progress.update(task, advance=1)
                continue
    
    if not dataframes:
        raise ValueError("No data files were successfully loaded!")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    console.print(f"[green]Loaded {len(combined_df)} rows from {len(dataframes)} files[/green]")
    
    return combined_df


def extract_cwru_metadata_from_filename(filename: str, config: Dict) -> Dict:
    """
    Extract metadata from CWRU filename.
    
    Pattern: {fault_type}{size}_{load}__X{number}_{sensor}_time.csv
    Example: B007_0__X118_DE_time.csv
    """
    metadata = {}
    metadata_config = config.get('schema', {}).get('metadata_from_filename', {})
    
    if not metadata_config:
        return metadata
    
    base_name = pathlib.Path(filename).stem
    
    # Extract fault type
    if metadata_config.get('fault_type', False):
        if base_name.startswith('B'):
            metadata['fault_type'] = 'ball_fault'
        elif base_name.startswith('IR'):
            metadata['fault_type'] = 'inner_race_fault'
        elif base_name.startswith('OR'):
            metadata['fault_type'] = 'outer_race_fault'
        elif 'normal' in base_name.lower() or 'baseline' in base_name.lower():
            metadata['fault_type'] = 'normal'
        else:
            metadata['fault_type'] = 'unknown'
    
    # Extract fault size (e.g., 007, 014, 021)
    if metadata_config.get('fault_size', False):
        match = re.search(r'(\d{3})', base_name)
        if match:
            metadata['fault_size_mils'] = int(match.group(1))
    
    # Extract load (e.g., 0, 2)
    if metadata_config.get('load_hp', False):
        match = re.search(r'_(\d+)__', base_name)
        if match:
            metadata['load_hp'] = int(match.group(1))
    
    # Extract sensor location (DE, FE, BA)
    if metadata_config.get('sensor_location', False):
        if '_DE_' in base_name:
            metadata['sensor_location'] = 'drive_end'
        elif '_FE_' in base_name:
            metadata['sensor_location'] = 'fan_end'
        elif '_BA_' in base_name:
            metadata['sensor_location'] = 'base'
        else:
            metadata['sensor_location'] = 'unknown'
    
    return metadata


def load_cwru_data(config: Dict) -> pd.DataFrame:
    """
    Load CWRU bearing data from CSV files.
    
    Files are organized in directories (normal_baseline, drive_end_12k, etc.)
    Each CSV file contains vibration signal data with index and value columns.
    """
    raw_input_dirs = config['paths'].get('raw_input_dirs', [])
    
    if isinstance(raw_input_dirs, str):
        raw_input_dirs = [raw_input_dirs]
    
    # Get column mapping
    column_map = config.get('schema', {}).get('column_map', {})
    
    console.print(f"[green]Loading CWRU data from {len(raw_input_dirs)} directories...[/green]")
    
    dataframes = []
    
    for input_dir in raw_input_dirs:
        input_path = pathlib.Path(input_dir)
        if not input_path.exists():
            console.print(f"[yellow]Warning: Directory not found: {input_path}[/yellow]")
            continue
        
        # Get all CSV files
        csv_files = list(input_path.glob('*.csv'))
        console.print(f"  Found {len(csv_files)} CSV files in {input_path.name}")
        
        for csv_file in tqdm(csv_files, desc=f"Loading from {input_path.name}", leave=False):
            try:
                # Load CSV
                df = pd.read_csv(csv_file)
                
                # Apply column mapping
                if column_map:
                    df = df.rename(columns=column_map)
                
                # Extract metadata from filename
                metadata = extract_cwru_metadata_from_filename(csv_file.name, config)
                
                # Add metadata as columns
                for key, value in metadata.items():
                    df[key] = value
                
                # Add file identifier
                df['source_file'] = csv_file.name
                df['source_directory'] = input_path.name
                
                # Ensure numeric columns are numeric
                numeric_cols = config.get('schema', {}).get('numeric_features', [])
                if numeric_cols:
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if df.empty:
                    console.print(f"[yellow]Warning: File {csv_file.name} is empty after loading[/yellow]")
                    continue
                
                dataframes.append(df)
                
            except Exception as e:
                console.print(f"[yellow]Warning: Failed to load {csv_file.name}: {e}[/yellow]")
                continue
    
    if not dataframes:
        raise ValueError("No data files were successfully loaded!")
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    console.print(f"[green]Loaded {len(combined_df)} rows from {len(dataframes)} files[/green]")
    
    return combined_df


def load_ai4i_data(config: Dict) -> pd.DataFrame:
    """
    Load AI4I dataset from single CSV file.
    
    This is tabular data with headers, so loading is straightforward.
    """
    raw_input_file = pathlib.Path(config['paths']['raw_input_file'])
    
    if not raw_input_file.exists():
        raise FileNotFoundError(f"Input file not found: {raw_input_file}")
    
    console.print(f"[green]Loading AI4I data from {raw_input_file}...[/green]")
    
    # Load CSV with headers
    df = pd.read_csv(raw_input_file)
    
    # Apply column mapping if provided
    column_map = config.get('schema', {}).get('column_map', {})
    if column_map:
        df = df.rename(columns=column_map)
    
    # Ensure numeric columns are numeric
    numeric_cols = config.get('schema', {}).get('numeric_features', [])
    if numeric_cols:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Ensure categorical columns are strings
    categorical_cols = config.get('schema', {}).get('categorical_features', [])
    if categorical_cols:
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)
    
    console.print(f"[green]Loaded {len(df)} rows[/green]")
    
    return df


def load_cmapss_data(config: Dict) -> pd.DataFrame:
    """
    Load CMAPSS (FD001-FD004) dataset from train/test files.
    
    Files are space-separated with no headers. Multiple columns including
    unit ID, cycle number, settings, and sensor readings.
    """
    # Get file paths
    raw_train_file = config['paths'].get('raw_train_file')
    raw_test_file = config['paths'].get('raw_test_file')
    raw_rul_file = config['paths'].get('raw_rul_file')
    
    # Get file format settings
    file_format = config.get('file_format', {})
    delimiter = file_format.get('delimiter', r'\s+')
    header = file_format.get('header', False)
    columns_in_order = file_format.get('columns_in_order', [])
    
    dataframes = []
    
    # Load training data
    if raw_train_file:
        train_path = pathlib.Path(raw_train_file)
        if train_path.exists():
            console.print(f"[green]Loading CMAPSS training data from {train_path}...[/green]")
            df_train = pd.read_csv(
                train_path,
                delimiter=delimiter,
                header=None,
                names=columns_in_order,
                engine='python'
            )
            df_train['split'] = 'train'
            dataframes.append(df_train)
            console.print(f"  Loaded {len(df_train)} rows")
        else:
            console.print(f"[yellow]Warning: Training file not found: {train_path}[/yellow]")
    
    # Load test data
    if raw_test_file:
        test_path = pathlib.Path(raw_test_file)
        if test_path.exists():
            console.print(f"[green]Loading CMAPSS test data from {test_path}...[/green]")
            df_test = pd.read_csv(
                test_path,
                delimiter=delimiter,
                header=None,
                names=columns_in_order,
                engine='python'
            )
            df_test['split'] = 'test'
            dataframes.append(df_test)
            console.print(f"  Loaded {len(df_test)} rows")
        else:
            console.print(f"[yellow]Warning: Test file not found: {test_path}[/yellow]")
    
    if not dataframes:
        raise ValueError("No CMAPSS data files were successfully loaded!")
    
    # Combine train and test
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    # Load RUL file if provided (for test set)
    # Note: RUL values will be used in feature engineering step, not here
    if raw_rul_file:
        rul_path = pathlib.Path(raw_rul_file)
        if rul_path.exists():
            console.print(f"[blue]Note: RUL file found at {rul_path} (will be used in feature engineering)[/blue]")
    
    # Ensure numeric columns are numeric
    numeric_cols = config.get('schema', {}).get('numeric_features', [])
    if numeric_cols:
        for col in numeric_cols:
            if col in combined_df.columns:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    console.print(f"[green]Loaded {len(combined_df)} total rows[/green]")
    
    return combined_df


def clean_data(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    Clean and validate data based on configuration.
    
    Performs:
    - Missing value handling (drop or impute)
    - Data type validation
    - Basic data quality checks
    """
    prep_config = config.get('prep', {})
    schema_config = config.get('schema', {})
    
    original_rows = len(df)
    
    # Handle missing values
    drop_na = prep_config.get('drop_na', True)
    if drop_na:
        df = df.dropna()
        console.print(f"[blue]Dropped rows with missing values: {original_rows} -> {len(df)} rows[/blue]")
    else:
        # Impute missing values
        impute_method = prep_config.get('impute', 'mean')
        numeric_cols = schema_config.get('numeric_features', [])
        
        if numeric_cols and impute_method:
            for col in numeric_cols:
                if col in df.columns:
                    if df[col].isna().any():
                        if impute_method == 'mean':
                            df[col].fillna(df[col].mean(), inplace=True)
                        elif impute_method == 'median':
                            df[col].fillna(df[col].median(), inplace=True)
                        elif impute_method == 'mode':
                            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0, inplace=True)
        
        console.print(f"[blue]Imputed missing values using {impute_method} method[/blue]")
    
    # Validate data types
    numeric_cols = schema_config.get('numeric_features', [])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            # Check for any remaining non-numeric values
            if df[col].isna().any():
                console.print(f"[yellow]Warning: Column {col} has non-numeric values after conversion[/yellow]")
    
    # Data quality checks
    console.print(f"[blue]Data quality checks:[/blue]")
    console.print(f"  Total rows: {len(df)}")
    console.print(f"  Total columns: {len(df.columns)}")
    console.print(f"  Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    return df


def save_cleaned_data(df: pd.DataFrame, config: Dict) -> None:
    """Save cleaned data to parquet format."""
    output_path = pathlib.Path(config['paths']['clean_output_path'])
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[green]Saving cleaned data to {output_path}...[/green]")
    
    # Save as parquet (efficient, compressed format)
    df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
    
    console.print(f"[green]Successfully saved {len(df)} rows to {output_path}[/green]")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Prepare and clean raw data for predictive maintenance pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file (e.g., configs/ims.yaml)'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    console.print(f"[cyan]Loading configuration from {args.config}...[/cyan]")
    config = load_config(args.config)
    
    dataset_name = config.get('dataset_name', 'unknown')
    console.print(f"[cyan]Dataset: {dataset_name}[/cyan]")
    
    # Load data based on dataset type
    # Determine dataset type by checking what paths are available
    paths = config.get('paths', {})
    
    if 'raw_input_dir' in paths:
        # IMS dataset (single directory with timestamped files)
        df = load_ims_data(config)
    elif 'raw_input_dirs' in paths:
        # CWRU dataset (multiple directories with CSV files)
        df = load_cwru_data(config)
    elif 'raw_input_file' in paths:
        # AI4I dataset (single CSV file)
        df = load_ai4i_data(config)
    elif 'raw_train_file' in paths or 'raw_test_file' in paths:
        # CMAPSS dataset (train/test files)
        df = load_cmapss_data(config)
    else:
        raise ValueError(
            "Could not determine dataset type from configuration. "
            "Expected one of: raw_input_dir, raw_input_dirs, raw_input_file, raw_train_file"
        )
    
    # Clean data
    df = clean_data(df, config)
    
    # Save cleaned data
    save_cleaned_data(df, config)
    
    # Final summary
    console.print("\n[bold cyan]Data Preparation Summary:[/bold cyan]")
    console.print(f"  Dataset: {dataset_name}")
    console.print(f"  Total rows: {len(df):,}")
    console.print(f"  Total columns: {len(df.columns)}")
    console.print(f"  Output file: {config['paths']['clean_output_path']}")
    console.print("[bold green]Data preparation complete![/bold green]")


if __name__ == '__main__':
    main()

