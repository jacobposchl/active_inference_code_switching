"""
Visualization Module
Creates plots and figures for model results
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path


def plot_model_comparison(results_dict, config, save_path=None):
    """
    Create comprehensive model comparison visualization
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping model names to results
    config : dict
        Visualization configuration
    save_path : str or None
        Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    colors = config.get('colors', {})
    
    models = list(results_dict.keys())
    model_colors = [colors.get(m, '#1f77b4') for m in models]
    
    # Plot 1: Log-likelihood comparison
    log_liks = [results_dict[m]['total_log_lik'] for m in models]
    axes[0, 0].bar(models, log_liks, color=model_colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Total Log-Likelihood', fontsize=12)
    axes[0, 0].set_title('Model Comparison\n(Higher is Better)', fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].tick_params(axis='x', rotation=15)
    
    # Plot 2: AIC comparison
    aics = [results_dict[m]['aic'] for m in models]
    axes[0, 1].bar(models, aics, color=model_colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('AIC', fontsize=12)
    axes[0, 1].set_title('AIC Comparison\n(Lower is Better)', fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].tick_params(axis='x', rotation=15)
    
    # Plot 3: Accuracy comparison
    accs = [results_dict[m]['accuracy'] for m in models]
    axes[0, 2].bar(models, accs, color=model_colors, alpha=0.7, edgecolor='black')
    axes[0, 2].axhline(0.5, color='red', linestyle='--', label='Chance', linewidth=2)
    axes[0, 2].set_ylabel('Accuracy', fontsize=12)
    axes[0, 2].set_title('Classification Accuracy\n(Higher is Better)', 
                         fontsize=13, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3, axis='y')
    axes[0, 2].tick_params(axis='x', rotation=15)
    
    # Plot 4: Log-likelihood distributions
    axes[1, 0].hist([results_dict[m]['log_likelihoods'] for m in models],
                    label=models, bins=50, alpha=0.6)
    axes[1, 0].set_xlabel('Log-Likelihood per sentence')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Log-Likelihood Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Gamma distributions for dynamic models
    dynamic_models = [m for m in models if 'M2' in m or 'M3' in m]
    if dynamic_models:
        for m in dynamic_models:
            axes[1, 1].hist(results_dict[m]['gammas'], bins=30, alpha=0.5, label=m)
        axes[1, 1].set_xlabel('Gamma (Precision)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Precision Distributions')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Prediction distributions
    axes[1, 2].hist([results_dict[m]['predictions'] for m in models],
                    label=models, bins=30, alpha=0.6)
    axes[1, 2].axvline(0.5, color='black', linestyle='--', label='Decision boundary')
    axes[1, 2].set_xlabel('P(switch)')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].set_title('Predicted Switch Probabilities')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.get('dpi', 300), bbox_inches='tight')
        print(f"\nSaved figure to {save_path}")
    
    plt.show()


def plot_m3_mechanism(results_m3, model_data, config, save_path=None):
    """
    Visualize M3 model mechanism
    
    Parameters:
    -----------
    results_m3 : dict
        M3 results dictionary
    model_data : DataFrame
        Model data
    config : dict
        Visualization configuration
    save_path : str or None
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    surprisal_vals = model_data['surprisal_first_cs_word_trans'].values
    high_load_probs = results_m3['inferred_states'][:, 1]
    gammas = results_m3['gammas']
    predictions = results_m3['predictions']
    actual = model_data['cs_binary'].values
    
    # Plot 1: State inference
    scatter = axes[0, 0].scatter(surprisal_vals, high_load_probs,
                                 c=actual, cmap='RdYlBu_r',
                                 alpha=0.4, s=20, edgecolors='none')
    axes[0, 0].set_xlabel('Word Surprisal', fontsize=12)
    axes[0, 0].set_ylabel('P(High-Load State)', fontsize=12)
    axes[0, 0].set_title('State Inference\n(Color: Red=Switch, Blue=No Switch)',
                         fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0, 0])
    
    # Plot 2: Precision modulation
    scatter2 = axes[0, 1].scatter(surprisal_vals, gammas,
                                  c=high_load_probs, cmap='viridis',
                                  alpha=0.4, s=20, edgecolors='none')
    axes[0, 1].set_xlabel('Word Surprisal', fontsize=12)
    axes[0, 1].set_ylabel('Gamma (Precision)', fontsize=12)
    axes[0, 1].set_title('Precision Modulation\n(Color: Belief in High-Load)',
                         fontsize=13, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[0, 1])
    
    # Plot 3: Switch predictions
    axes[1, 0].scatter(surprisal_vals, predictions,
                       c=actual, cmap='RdYlBu_r',
                       alpha=0.4, s=20, edgecolors='none')
    axes[1, 0].axhline(0.5, color='black', linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel('Word Surprisal', fontsize=12)
    axes[1, 0].set_ylabel('P(Code-Switch)', fontsize=12)
    axes[1, 0].set_title('Model Predictions\n(Color: Actual Outcome)',
                         fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Predictions vs Actual by Surprisal Bin
    model_data_copy = model_data.copy()
    model_data_copy['m3_pred'] = predictions
    
    surprisal_bins = ['Low', 'Medium', 'High']
    predicted_by_bin = [
        model_data_copy[model_data_copy['surprisal_idx']==0]['m3_pred'].mean(),
        model_data_copy[model_data_copy['surprisal_idx']==1]['m3_pred'].mean(),
        model_data_copy[model_data_copy['surprisal_idx']==2]['m3_pred'].mean()
    ]
    actual_by_bin = [
        model_data_copy[model_data_copy['surprisal_idx']==0]['cs_binary'].mean(),
        model_data_copy[model_data_copy['surprisal_idx']==1]['cs_binary'].mean(),
        model_data_copy[model_data_copy['surprisal_idx']==2]['cs_binary'].mean()
    ]
    
    x = np.arange(len(surprisal_bins))
    width = 0.35
    axes[1, 1].bar(x - width/2, actual_by_bin, width, label='Actual',
                   color='steelblue', alpha=0.8, edgecolor='black')
    axes[1, 1].bar(x + width/2, predicted_by_bin, width, label='M3 Predicted',
                   color='coral', alpha=0.8, edgecolor='black')
    axes[1, 1].set_xlabel('Surprisal Level', fontsize=12)
    axes[1, 1].set_ylabel('Code-Switch Rate', fontsize=12)
    axes[1, 1].set_title('Predicted vs Actual Switch Rates', 
                         fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(surprisal_bins)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=config.get('dpi', 300), bbox_inches='tight')
        print(f"\nSaved figure to {save_path}")
    
    plt.show()


def plot_pair_analysis(pairs_data, save_path=None):
    """
    Visualize within-pair analysis
    
    Parameters:
    -----------
    pairs_data : DataFrame
        Pair-level data with differences
    save_path : str or None
        Path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Distribution of surprisal differences
    axes[0, 0].hist(pairs_data['surprisal_diff'], bins=50, color='steelblue',
                    alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(0, color='red', linestyle='--', linewidth=2, label='No difference')
    axes[0, 0].axvline(pairs_data['surprisal_diff'].mean(), color='green',
                       linestyle='--', linewidth=2, 
                       label=f'Mean = {pairs_data["surprisal_diff"].mean():.3f}')
    axes[0, 0].set_xlabel('Surprisal Difference (CS - non-CS)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title('Distribution of Within-Pair Surprisal Differences',
                         fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Scatter plot of CS vs non-CS surprisal
    axes[0, 1].scatter(pairs_data['surprisal_first_cs_word_trans_False'],
                       pairs_data['surprisal_first_cs_word_trans_True'],
                       alpha=0.4, s=20, color='steelblue')
    min_val = min(pairs_data['surprisal_first_cs_word_trans_False'].min(),
                  pairs_data['surprisal_first_cs_word_trans_True'].min())
    max_val = max(pairs_data['surprisal_first_cs_word_trans_False'].max(),
                  pairs_data['surprisal_first_cs_word_trans_True'].max())
    axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--',
                    linewidth=2, label='Equal surprisal')
    axes[0, 1].set_xlabel('Non-CS Surprisal', fontsize=12)
    axes[0, 1].set_ylabel('CS Surprisal', fontsize=12)
    axes[0, 1].set_title('CS vs Non-CS Surprisal (Each Point = 1 Pair)',
                         fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Length differences
    axes[1, 0].hist(pairs_data['length_diff'], bins=50, color='coral',
                    alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].axvline(pairs_data['length_diff'].mean(), color='green',
                       linestyle='--', linewidth=2, 
                       label=f'Mean = {pairs_data["length_diff"].mean():.3f}')
    axes[1, 0].set_xlabel('Length Difference (CS - non-CS)', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Distribution of Within-Pair Length Differences',
                         fontsize=13, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Relationship between surprisal diff and length diff
    axes[1, 1].scatter(pairs_data['surprisal_diff'], pairs_data['length_diff'],
                       alpha=0.4, s=20, color='mediumpurple')
    axes[1, 1].axhline(0, color='gray', linestyle='--', linewidth=1)
    axes[1, 1].axvline(0, color='gray', linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Surprisal Difference', fontsize=12)
    axes[1, 1].set_ylabel('Length Difference', fontsize=12)
    axes[1, 1].set_title('Surprisal vs Length Differences',
                         fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    corr = pairs_data['surprisal_diff'].corr(pairs_data['length_diff'])
    axes[1, 1].text(0.05, 0.95, f'Correlation: {corr:.3f}',
                    transform=axes[1, 1].transAxes, fontsize=11,
                    verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved figure to {save_path}")
    
    plt.show()
