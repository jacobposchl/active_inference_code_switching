"""
Model Comparison Script
Compare Active Inference models against logistic regression baselines
"""

import sys
from pathlib import Path
import pickle
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import VIZ_CONFIG
from src.evaluation.visualization import plot_model_comparison


def load_results(results_file='results/training_results.pkl'):
    """Load saved training results"""
    with open(results_file, 'rb') as f:
        results = pickle.load(f)
    return results


def print_detailed_comparison(all_results):
    """
    Print detailed comparison table
    
    Parameters:
    -----------
    all_results : dict
        Dictionary mapping model names to results
    """
    print("\n" + "="*80)
    print("DETAILED MODEL COMPARISON")
    print("="*80)
    
    # Organize models by type
    ai_models = [m for m in all_results.keys() if m.startswith('M')]
    baseline_models = [m for m in all_results.keys() if m.startswith('LR')]
    
    print(f"\n{'Model':<15s} {'Accuracy':>10s} {'Log-Lik':>10s} {'AIC':>10s} {'BIC':>10s} {'N Params':>10s}")
    print("-"*80)
    
    # Print Active Inference models
    print("\nActive Inference Models:")
    print("-"*80)
    for model in ai_models:
        r = all_results[model]
        print(f"{model:<15s} {r['accuracy']:>10.4f} {r['total_log_lik']:>10.2f} "
              f"{r['aic']:>10.2f} {r['bic']:>10.2f} {r['n_params']:>10d}")
    
    # Print Baseline models
    if baseline_models:
        print("\nLogistic Regression Baselines:")
        print("-"*80)
        for model in sorted(baseline_models):
            r = all_results[model]
            print(f"{model:<15s} {r['accuracy']:>10.4f} {r['total_log_lik']:>10.2f} "
                  f"{r['aic']:>10.2f} {r['bic']:>10.2f} {r['n_params']:>10d}")
    
    # Find best models
    print("\n" + "="*80)
    print("BEST MODELS")
    print("="*80)
    
    best_acc = max(all_results.items(), key=lambda x: x[1]['accuracy'])
    best_ll = max(all_results.items(), key=lambda x: x[1]['total_log_lik'])
    best_aic = min(all_results.items(), key=lambda x: x[1]['aic'])
    best_bic = min(all_results.items(), key=lambda x: x[1]['bic'])
    
    print(f"Best Accuracy:       {best_acc[0]:<15s} ({best_acc[1]['accuracy']:.4f})")
    print(f"Best Log-Likelihood: {best_ll[0]:<15s} ({best_ll[1]['total_log_lik']:.2f})")
    print(f"Best AIC:            {best_aic[0]:<15s} ({best_aic[1]['aic']:.2f})")
    print(f"Best BIC:            {best_bic[0]:<15s} ({best_bic[1]['bic']:.2f})")


def compare_ai_vs_baselines(all_results):
    """
    Statistical comparison between AI models and baselines
    
    Parameters:
    -----------
    all_results : dict
        Dictionary mapping model names to results
    """
    print("\n" + "="*80)
    print("ACTIVE INFERENCE vs BASELINES")
    print("="*80)
    
    # Get best AI model and best baseline
    ai_models = {k: v for k, v in all_results.items() if k.startswith('M')}
    baseline_models = {k: v for k, v in all_results.items() if k.startswith('LR')}
    
    if not baseline_models:
        print("No baseline models found!")
        return
    
    best_ai = max(ai_models.items(), key=lambda x: x[1]['total_log_lik'])
    best_baseline = max(baseline_models.items(), key=lambda x: x[1]['total_log_lik'])
    
    print(f"\nBest Active Inference Model: {best_ai[0]}")
    print(f"Best Baseline Model:          {best_baseline[0]}")
    
    print(f"\n{'Metric':<20s} {best_ai[0]:>15s} {best_baseline[0]:>15s} {'Difference':>15s}")
    print("-"*80)
    
    acc_diff = best_ai[1]['accuracy'] - best_baseline[1]['accuracy']
    ll_diff = best_ai[1]['total_log_lik'] - best_baseline[1]['total_log_lik']
    aic_diff = best_ai[1]['aic'] - best_baseline[1]['aic']
    bic_diff = best_ai[1]['bic'] - best_baseline[1]['bic']
    
    print(f"{'Accuracy':<20s} {best_ai[1]['accuracy']:>15.4f} {best_baseline[1]['accuracy']:>15.4f} "
          f"{acc_diff:>+15.4f}")
    print(f"{'Log-Likelihood':<20s} {best_ai[1]['total_log_lik']:>15.2f} "
          f"{best_baseline[1]['total_log_lik']:>15.2f} {ll_diff:>+15.2f}")
    print(f"{'AIC (lower better)':<20s} {best_ai[1]['aic']:>15.2f} {best_baseline[1]['aic']:>15.2f} "
          f"{aic_diff:>+15.2f}")
    print(f"{'BIC (lower better)':<20s} {best_ai[1]['bic']:>15.2f} {best_baseline[1]['bic']:>15.2f} "
          f"{bic_diff:>+15.2f}")
    
    # Interpretation
    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    
    if ll_diff > 0:
        print(f"✓ Active Inference ({best_ai[0]}) has better log-likelihood (+{ll_diff:.2f})")
    else:
        print(f"✗ Baseline ({best_baseline[0]}) has better log-likelihood ({ll_diff:.2f})")
    
    if aic_diff < 0:
        print(f"✓ Active Inference has better (lower) AIC ({aic_diff:.2f})")
    else:
        print(f"✗ Baseline has better (lower) AIC (+{aic_diff:.2f})")
    
    if bic_diff < 0:
        print(f"✓ Active Inference has better (lower) BIC ({bic_diff:.2f})")
    else:
        print(f"✗ Baseline has better (lower) BIC (+{bic_diff:.2f})")
    
    # Overall verdict
    wins = sum([ll_diff > 0, aic_diff < 0, bic_diff < 0])
    print(f"\nActive Inference wins {wins}/3 metrics")


def create_comparison_table(all_results, save_path='results/model_comparison.csv'):
    """
    Create a CSV table of all results
    
    Parameters:
    -----------
    all_results : dict
        Dictionary mapping model names to results
    save_path : str
        Path to save CSV
    """
    rows = []
    for model_name, results in all_results.items():
        rows.append({
            'Model': model_name,
            'Type': 'Active Inference' if model_name.startswith('M') else 'Baseline',
            'Accuracy': results['accuracy'],
            'Log_Likelihood': results['total_log_lik'],
            'AIC': results['aic'],
            'BIC': results['bic'],
            'N_Params': results['n_params']
        })
    
    df = pd.DataFrame(rows)
    df = df.sort_values('Log_Likelihood', ascending=False)
    df.to_csv(save_path, index=False)
    print(f"\nSaved comparison table to {save_path}")
    
    return df


def main():
    """Main comparison script"""
    
    print("="*80)
    print("MODEL COMPARISON: ACTIVE INFERENCE vs LOGISTIC REGRESSION")
    print("="*80)
    
    # Load results
    try:
        results = load_results()
        print("\n✓ Loaded results from results/training_results.pkl")
    except FileNotFoundError:
        print("\n✗ ERROR: results/training_results.pkl not found!")
        print("Please run train_models.py first")
        return
    
    all_results = results['all_results']
    
    # Print detailed comparison
    print_detailed_comparison(all_results)
    
    # Compare AI vs baselines
    compare_ai_vs_baselines(all_results)
    
    # Create CSV table
    df = create_comparison_table(all_results)
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING PLOTS")
    print("="*80)
    
    from pathlib import Path
    results_dir = VIZ_CONFIG['results_dir']
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Model comparison plot
    print("\nGenerating model comparison plot...")
    plot_model_comparison(all_results, VIZ_CONFIG, 
                         save_path=f'{results_dir}/model_comparison.png')
    
    print("✓ Saved to results/model_comparison.png")
    
    # Print cross-validation summary if available
    if 'cv_summary' in results:
        print("\n" + "="*80)
        print("CROSS-VALIDATION SUMMARY (Mean ± Std)")
        print("="*80)
        cv_summary = results['cv_summary']
        
        print(f"\n{'Model':<15s} {'CV Accuracy':>20s} {'CV Log-Lik':>20s}")
        print("-"*80)
        for model, summary in cv_summary.items():
            if model in all_results:  # Only show if in final results
                print(f"{model:<15s} "
                      f"{summary['mean_accuracy']:>8.4f} ± {summary['std_accuracy']:<8.4f} "
                      f"{summary['mean_loglik']:>8.2f} ± {summary['std_loglik']:<8.2f}")
    
    print("\n" + "="*80)
    print("COMPARISON COMPLETE!")
    print("="*80)
    print("\nFiles generated:")
    print("  - results/model_comparison.csv")
    print("  - results/model_comparison.png")
    
    # Check if plots exist
    if Path('results/m3_mechanism.png').exists():
        print("  - results/m3_mechanism.png (from training)")
    else:
        print("  - results/m3_mechanism.png (not found - run train_models.py to generate)")
    
    print("\nTo view plots:")
    print("  Open results/model_comparison.png in your image viewer")


if __name__ == '__main__':
    main()
