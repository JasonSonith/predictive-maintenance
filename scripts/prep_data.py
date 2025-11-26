import argparse
import pathlib
import re
from datetime import datetime

import pandas as pd
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_timestamp(filename, timestamp_format):
    base_name = pathlib.Path(filename).stem
    try:
        return datetime.strptime(base_name, timestamp_format)
    except ValueError:
        # Try without seconds if the format includes seconds
        if '%S' in timestamp_format:
            alt_format = timestamp_format.replace('.%S', '').replace('%S', '')
            try:
                return datetime.strptime(base_name, alt_format)
            except ValueError:
                pass
        return None

def load_ims_data(config):
    raw_input_dir = pathlib.Path(config['paths']['raw_input_dir'])
    max_files = config['paths'].get('max_files')
    
    file_format = config.get('file_format', {})
    delimiter = file_format.get('delimiter', '\t')
    columns_in_order = file_format.get('columns_in_order', [])
    extract_timestamp_flag = file_format.get('extract_timestamp_from_filename', False)
    timestamp_format = file_format.get('timestamp_format', '%Y.%m.%d.%H.%M.%S')
    column_map = config.get('schema', {}).get('column_map', {})
    
    all_files = sorted([f for f in raw_input_dir.iterdir() if f.is_file()])
    if max_files:
        all_files = all_files[:max_files]
    
    print(f"Loading {len(all_files)} files...")
    dataframes = []
    
    for file_path in all_files:
        df = pd.read_csv(file_path, delimiter=delimiter, header=None, names=columns_in_order)
        
        if column_map and not columns_in_order:
            df.columns = [column_map.get(str(i), f'col_{i}') for i in range(len(df.columns))]
        elif columns_in_order:
            df.columns = columns_in_order
        
        if extract_timestamp_flag:
            timestamp = extract_timestamp(file_path.name, timestamp_format)
            if timestamp:
                df['timestamp'] = timestamp
            df['file_index'] = len(dataframes)
        
        df['source_file'] = file_path.name
        dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows")
    return combined_df

def extract_cwru_metadata(filename, directory_name, config):
    metadata = {}
    metadata_config = config.get('schema', {}).get('metadata_from_filename', {})
    if not metadata_config:
        return metadata

    base_name = pathlib.Path(filename).stem

    if metadata_config.get('fault_type', False):
        if base_name.startswith('B'):
            metadata['fault_type'] = 'ball_fault'
        elif base_name.startswith('IR'):
            metadata['fault_type'] = 'inner_race_fault'
        elif base_name.startswith('OR'):
            metadata['fault_type'] = 'outer_race_fault'
        elif 'normal' in base_name.lower() or 'baseline' in base_name.lower() or 'normal' in directory_name.lower():
            metadata['fault_type'] = 'normal'
        else:
            metadata['fault_type'] = 'unknown'
    
    if metadata_config.get('fault_size', False):
        match = re.search(r'(\d{3})', base_name)
        if match:
            metadata['fault_size_mils'] = int(match.group(1))
        else:
            metadata['fault_size_mils'] = None

    if metadata_config.get('load_hp', False):
        match = re.search(r'_(\d+)__', base_name)
        if match:
            metadata['load_hp'] = int(match.group(1))
        else:
            # Default to 0 hp load for files without explicit load specification (e.g., normal baseline)
            metadata['load_hp'] = 0
    
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

def load_cwru_data(config):
    raw_input_dirs = config['paths'].get('raw_input_dirs', [])
    if isinstance(raw_input_dirs, str):
        raw_input_dirs = [raw_input_dirs]
    
    column_map = config.get('schema', {}).get('column_map', {})
    dataframes = []
    
    for input_dir in raw_input_dirs:
        input_path = pathlib.Path(input_dir)
        csv_files = list(input_path.glob('*.csv'))
        print(f"Loading {len(csv_files)} files from {input_path.name}")
        
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)

            if column_map:
                df = df.rename(columns=column_map)

            metadata = extract_cwru_metadata(csv_file.name, input_path.name, config)
            for key, value in metadata.items():
                df[key] = value

            df['source_file'] = csv_file.name
            df['source_directory'] = input_path.name
            dataframes.append(df)
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Loaded {len(combined_df)} rows")
    return combined_df

def load_ai4i_data(config):
    raw_input_file = pathlib.Path(config['paths']['raw_input_file'])
    print(f"Loading {raw_input_file}...")
    
    df = pd.read_csv(raw_input_file)
    
    column_map = config.get('schema', {}).get('column_map', {})
    if column_map:
        df = df.rename(columns=column_map)
    
    print(f"Loaded {len(df)} rows")
    return df

def load_cmapss_data(config):
    raw_train_file = config['paths'].get('raw_train_file')
    raw_test_file = config['paths'].get('raw_test_file')
    
    file_format = config.get('file_format', {})
    delimiter = file_format.get('delimiter', r'\s+')
    columns_in_order = file_format.get('columns_in_order', [])
    
    dataframes = []
    
    if raw_train_file:
        train_path = pathlib.Path(raw_train_file)
        print(f"Loading training data from {train_path}...")
        df_train = pd.read_csv(train_path, delimiter=delimiter, header=None, names=columns_in_order, engine='python')
        df_train['split'] = 'train'
        dataframes.append(df_train)
        print(f"Loaded {len(df_train)} rows")
    
    if raw_test_file:
        test_path = pathlib.Path(raw_test_file)
        print(f"Loading test data from {test_path}...")
        df_test = pd.read_csv(test_path, delimiter=delimiter, header=None, names=columns_in_order, engine='python')
        df_test['split'] = 'test'
        dataframes.append(df_test)
        print(f"Loaded {len(df_test)} rows")
    
    combined_df = pd.concat(dataframes, ignore_index=True)
    print(f"Total rows: {len(combined_df)}")
    return combined_df

def clean_data(df, config):
    prep_config = config.get('prep', {})
    schema_config = config.get('schema', {})

    drop_na = prep_config.get('drop_na', True)
    if drop_na:
        print(f"Columns with NaN values before dropping:")
        for col in df.columns:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                print(f"  {col}: {nan_count} NaN values ({100*nan_count/len(df):.1f}%)")
        df = df.dropna()
        print(f"Dropped missing values: {len(df)} rows remaining")
    else:
        impute_method = prep_config.get('impute', 'mean')
        numeric_cols = schema_config.get('numeric_features', [])
        
        for col in numeric_cols:
            if col in df.columns and df[col].isna().any():
                if impute_method == 'mean':
                    df[col].fillna(df[col].mean(), inplace=True)
                elif impute_method == 'median':
                    df[col].fillna(df[col].median(), inplace=True)
                elif impute_method == 'mode':
                    df[col].fillna(df[col].mode()[0], inplace=True)
    
    numeric_cols = schema_config.get('numeric_features', [])
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    categorical_cols = schema_config.get('categorical_features', [])
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df

def save_cleaned_data(df, config):
    paths = config.get('paths', {})
    
    # Save parquet
    if 'clean_output_path' in paths:
        output_path = pathlib.Path(paths['clean_output_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {output_path} (Parquet)...")
        df.to_parquet(output_path, index=False, engine='pyarrow', compression='snappy')
        print(f"Saved {len(df)} rows to parquet")
    
    # Save csv
    if 'clean_output_path_csv' in paths:
        csv_path = pathlib.Path(paths['clean_output_path_csv'])
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving to {csv_path} (CSV)...")
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(df)} rows to csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    dataset_name = config.get('dataset_name', 'unknown')
    print(f"Dataset: {dataset_name}")
    
    paths = config.get('paths', {})
    
    if 'raw_input_dir' in paths:
        df = load_ims_data(config)
    elif 'raw_input_dirs' in paths:
        df = load_cwru_data(config)
    elif 'raw_input_file' in paths:
        df = load_ai4i_data(config)
    elif 'raw_train_file' in paths or 'raw_test_file' in paths:
        df = load_cmapss_data(config)
    else:
        raise ValueError("Unknown dataset type")
    
    df = clean_data(df, config)
    save_cleaned_data(df, config)
    
    print("Done!")

if __name__ == '__main__':
    main()