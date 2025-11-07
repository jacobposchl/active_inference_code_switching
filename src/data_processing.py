"""
Data Processing Module
Handles data loading, cleaning, discretization, and preparation
"""

import pandas as pd
import numpy as np
from scipy import stats
import pickle
from pathlib import Path


def load_raw_data(filepath):
    """
    Load raw code-switching data from CSV
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
        
    Returns:
    --------
    data : pd.DataFrame
        Raw data
    """
    data = pd.read_csv(filepath)
    print(f"Loaded {len(data)} sentences from {filepath}")
    return data


def clean_data(data, required_columns):
    """
    Remove rows with missing values in key columns
    
    Parameters:
    -----------
    data : pd.DataFrame
        Raw data
    required_columns : list
        List of column names that must be non-null
        
    Returns:
    --------
    data_clean : pd.DataFrame
        Cleaned data with added features
    """
    # Create binary target
    data['cs_binary'] = (data['sent_type'] == 'code-switch').astype(int)
    
    # Remove missing values
    data_clean = data.dropna(subset=required_columns)
    
    removed = len(data) - len(data_clean)
    print(f"Removed {removed} rows with missing values ({removed/len(data)*100:.1f}%)")
    print(f"Clean dataset size: {len(data_clean)}")
    
    return data_clean


def create_matched_pairs(data):
    """
    Create pair IDs for matched sentence pairs
    
    Parameters:
    -----------
    data : pd.DataFrame
        Cleaned data
        
    Returns:
    --------
    data : pd.DataFrame
        Data with pair_id and is_cs columns added
    """
    # Assuming alternating structure: 0&1 are pair, 2&3 are pair, etc.
    data['pair_id'] = data.index // 2
    data['is_cs'] = data['sent_type'] == 'code-switch'
    
    # Verify pairing structure
    pair_cs_counts = data.groupby('pair_id')['is_cs'].sum()
    valid_pairs = (pair_cs_counts == 1).sum()
    total_pairs = data['pair_id'].nunique()
    
    print(f"\nPair Structure:")
    print(f"  Total pairs: {total_pairs}")
    print(f"  Valid pairs (1 CS per pair): {valid_pairs}")
    print(f"  Invalid pairs: {total_pairs - valid_pairs}")
    
    if valid_pairs == total_pairs:
        print("  ✓ Pairing structure verified!")
    else:
        print("  ⚠️ Warning: Some pairs don't have exactly 1 code-switched sentence")
    
    return data


def discretize_variables(data, n_bins_surprisal=3, n_bins_length=3, n_bins_frequency=3):
    """
    Discretize continuous variables into bins
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with continuous variables
    n_bins_surprisal : int
        Number of bins for surprisal
    n_bins_length : int
        Number of bins for length
    n_bins_frequency : int
        Number of bins for frequency
        
    Returns:
    --------
    data : pd.DataFrame
        Data with binned variables added
    """
    # Discretize surprisal using quantiles
    data['surprisal_idx'] = pd.qcut(
        data['surprisal_first_cs_word_trans'],
        q=n_bins_surprisal,
        labels=False,
        duplicates='drop'
    )
    
    # Discretize length using quantiles
    data['length_idx'] = pd.qcut(
        data['translation_sentence_length'],
        q=n_bins_length,
        labels=False,
        duplicates='drop'
    )
    
    # Create human-readable labels for inspection
    surprisal_labels = [f"surp_bin_{i}" for i in range(n_bins_surprisal)]
    length_labels = [f"len_bin_{i}" for i in range(n_bins_length)]
    
    data['surprisal_binned'] = pd.qcut(
        data['surprisal_first_cs_word_trans'],
        q=n_bins_surprisal,
        labels=surprisal_labels,
        duplicates='drop'
    )
    
    data['length_binned'] = pd.qcut(
        data['translation_sentence_length'],
        q=n_bins_length,
        labels=length_labels,
        duplicates='drop'
    )
    
    print("\nDiscretization Summary:")
    print(f"  Surprisal: {n_bins_surprisal} bins")
    print(f"    Distribution: {data['surprisal_binned'].value_counts().sort_index().to_dict()}")
    print(f"  Length: {n_bins_length} bins")
    print(f"    Distribution: {data['length_binned'].value_counts().sort_index().to_dict()}")
    
    return data


def prepare_model_data(data):
    """
    Extract relevant columns for modeling
    
    Parameters:
    -----------
    data : pd.DataFrame
        Full processed data
        
    Returns:
    --------
    model_data : pd.DataFrame
        Subset of data with modeling columns
    """
    model_data = data[[
        'cs_binary', 
        'surprisal_idx', 
        'length_idx',
        'surprisal_first_cs_word_trans',
        'translation_sentence_length',
        'pair_id', 
        'is_cs'
    ]].copy()
    
    print(f"\nFinal modeling dataset: {len(model_data)} observations")
    return model_data


def compute_pair_statistics(data):
    """
    Compute within-pair differences and statistics
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data with pair information
        
    Returns:
    --------
    pairs : pd.DataFrame
        Pair-level data with differences
    stats_dict : dict
        Summary statistics
    """
    # Reshape to pair-level data
    pairs = data.pivot_table(
        index='pair_id',
        columns='is_cs',
        values=['surprisal_first_cs_word_trans',
                'translation_sentence_length',
                'entropy_at_cs_point'],
        aggfunc='first'
    )
    
    # Flatten column names
    pairs.columns = ['_'.join([str(c) for c in col]).strip() 
                     for col in pairs.columns.values]
    pairs = pairs.reset_index()
    
    # Compute within-pair differences (CS - non-CS)
    pairs['surprisal_diff'] = (
        pairs['surprisal_first_cs_word_trans_True'] -
        pairs['surprisal_first_cs_word_trans_False']
    )
    pairs['length_diff'] = (
        pairs['translation_sentence_length_True'] -
        pairs['translation_sentence_length_False']
    )
    pairs['entropy_diff'] = (
        pairs['entropy_at_cs_point_True'] -
        pairs['entropy_at_cs_point_False']
    )
    
    # Test hypothesis
    pairs['cs_has_higher_surprisal'] = pairs['surprisal_diff'] > 0
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(
        pairs['surprisal_first_cs_word_trans_True'],
        pairs['surprisal_first_cs_word_trans_False']
    )
    
    stats_dict = {
        'n_pairs': len(pairs),
        'prop_cs_higher_surprisal': pairs['cs_has_higher_surprisal'].mean(),
        'mean_surprisal_diff': pairs['surprisal_diff'].mean(),
        'std_surprisal_diff': pairs['surprisal_diff'].std(),
        'cohens_d': pairs['surprisal_diff'].mean() / pairs['surprisal_diff'].std(),
        't_statistic': t_stat,
        'p_value': p_value
    }
    
    return pairs, stats_dict


def save_processed_data(model_data, filepath):
    """
    Save processed data to disk
    
    Parameters:
    -----------
    model_data : pd.DataFrame
        Processed data
    filepath : str
        Output file path
    """
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    model_data.to_pickle(filepath)
    print(f"\nSaved processed data to {filepath}")


def load_processed_data(filepath):
    """
    Load processed data from disk
    
    Parameters:
    -----------
    filepath : str
        Path to processed data file
        
    Returns:
    --------
    model_data : pd.DataFrame
        Processed data
    """
    model_data = pd.read_pickle(filepath)
    print(f"Loaded processed data from {filepath}")
    print(f"  Shape: {model_data.shape}")
    return model_data


def print_data_summary(data):
    """
    Print comprehensive data summary
    
    Parameters:
    -----------
    data : pd.DataFrame
        Data to summarize
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    print(f"\nTotal sentences: {len(data)}")
    print(f"Code-switch rate: {data['cs_binary'].mean():.3f}")
    
    print("\nMean values by code-switch status:")
    for col in ['surprisal_first_cs_word_trans', 'translation_sentence_length', 
                'entropy_at_cs_point']:
        if col in data.columns:
            cs_mean = data[data['cs_binary']==1][col].mean()
            non_cs_mean = data[data['cs_binary']==0][col].mean()
            diff = cs_mean - non_cs_mean
            print(f"  {col}:")
            print(f"    CS: {cs_mean:.3f}, Non-CS: {non_cs_mean:.3f}, Diff: {diff:.3f}")
    
    if 'surprisal_binned' in data.columns:
        print("\nCode-switch rate by surprisal bin:")
        for level in ['low', 'medium', 'high']:
            rate = data[data['surprisal_binned']==level]['cs_binary'].mean()
            count = (data['surprisal_binned']==level).sum()
            print(f"  {level:6s}: {rate:.3f} ({count} sentences)")


def full_preprocessing_pipeline(raw_data_path, config):
    """
    Complete preprocessing pipeline
    
    Parameters:
    -----------
    raw_data_path : str
        Path to raw CSV file
    config : dict
        Configuration dictionary with processing parameters
        
    Returns:
    --------
    model_data : pd.DataFrame
        Processed and ready-to-use data
    pairs : pd.DataFrame
        Pair-level data with statistics
    stats_dict : dict
        Summary statistics
    """
    print("="*60)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    # Load and clean
    data = load_raw_data(raw_data_path)
    data = clean_data(data, config['required_columns'])
    
    # Create pairs
    data = create_matched_pairs(data)
    
    # Compute pair statistics
    pairs, stats_dict = compute_pair_statistics(data)
    
    print("\n" + "="*60)
    print("PAIR ANALYSIS RESULTS")
    print("="*60)
    print(f"Pairs where CS has higher surprisal: {stats_dict['prop_cs_higher_surprisal']:.3f}")
    print(f"Mean surprisal difference: {stats_dict['mean_surprisal_diff']:.3f}")
    print(f"Cohen's d: {stats_dict['cohens_d']:.3f}")
    print(f"t-statistic: {stats_dict['t_statistic']:.3f}, p-value: {stats_dict['p_value']:.4e}")
    
    if stats_dict['p_value'] < 0.001:
        print("*** Highly significant difference! ***")
    
    # Discretize
    data = discretize_variables(
        data,
        n_bins_surprisal=config['n_bins_surprisal'],
        n_bins_length=config['n_bins_length'],
        n_bins_frequency=config['n_bins_frequency']
    )
    
    # Prepare final dataset
    model_data = prepare_model_data(data)
    
    # Print summary
    print_data_summary(model_data)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    
    return model_data, pairs, stats_dict
