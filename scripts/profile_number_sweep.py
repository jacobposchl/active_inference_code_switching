"""
K-Profiles Experiment
Tests M3 with different numbers of profiles (k=1,2,3,4,5) to determine optimal k
"""

import sys
from pathlib import Path
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import *
from src.data_processing import load_processed_data
from src.models.active_inference import initialize_model
from src.models.value_functions import make_value_fn_M3
from src.models.profiles import create_profile_manager
from src.training.cross_validation import cross_validate_models
from src.training.optimizer import fit_M3_parameters
from pymdp.agent import Agent
from pymdp import control, utils


def create_k_profile_config(k, base_config=M3_CONFIG, arch_config=ARCHITECTURE_CONFIG):
    """
    Create M3 configuration with k profiles
    
    Parameters:
    -----------
    k : int
        Number of profiles
    base_config : dict
        Base M3 configuration
    arch_config : dict
        Architecture configuration
        
    Returns:
    --------
    config : dict
        Modified M3 config with k profiles
    """
    config = base_config.copy()
    config['num_profiles'] = k
    
    # Generate initial profiles
    # Strategy: Spread profiles across the parameter space
    initial_profiles = []
    
    num_states = arch_config['num_states']
    
    for i in range(k):
        # Create gradient from high-to-low preference for no-switch/switch
        # Profile 0: strong no-switch, Profile k-1: strong switch
        if k == 1:
            # Single profile: neutral
            phi_logit_0 = 0.0
            phi_logit_1 = 0.0
        else:
            # Interpolate from no-switch to switch preference
            t = i / (k - 1)  # 0 to 1
            phi_logit_0 = 2.0 - 4.0 * t  # 2.0 → -2.0
            phi_logit_1 = -2.0 + 4.0 * t  # -2.0 → 2.0
        
        # Similar for policy priors
        if k == 1:
            xi_logit_0 = 0.0
            xi_logit_1 = 0.0
        else:
            t = i / (k - 1)
            xi_logit_0 = 1.0 - 2.0 * t  # 1.0 → -1.0
            xi_logit_1 = -1.0 + 2.0 * t  # -1.0 → 1.0
        
        # Gamma: high precision for extreme profiles, lower for middle
        if k == 1:
            gamma = 1.5
        else:
            # U-shaped: high at extremes, low in middle
            distance_from_center = abs(2 * t - 1)  # 0 at center, 1 at extremes
            gamma = 1.0 + 2.0 * distance_from_center  # 1.0 → 3.0
        
        profile = {
            'phi_logits': np.array([phi_logit_0, phi_logit_1]),
            'xi_logits': np.array([xi_logit_0, xi_logit_1]),
            'gamma': gamma
        }
        initial_profiles.append(profile)
    
    config['initial_profiles'] = initial_profiles
    
    # Generate initial Z matrix: states map to nearest profiles
    if k == 1:
        # All states map to single profile
        Z = np.ones((num_states, 1))
    else:
        # Each state maps strongly to one profile
        Z = np.zeros((num_states, k))
        for s in range(num_states):
            # Assign state to profile with soft assignment
            # State 0 → Profile 0, State (num_states-1) → Profile (k-1)
            if num_states == 1:
                primary_profile = 0
            else:
                primary_profile = int(s * (k - 1) / (num_states - 1))
            
            # 80% to primary, 20% distributed to others
            Z[s, primary_profile] = 0.8
            if k > 1:
                for p in range(k):
                    if p != primary_profile:
                        Z[s, p] = 0.2 / (k - 1)
    
    config['initial_Z'] = Z
    
    return config


def compute_model_params_count(k, arch_config, learn_Z=False):
    """
    Compute number of parameters for M3 with k profiles
    
    Parameters:
    -----------
    k : int
        Number of profiles
    arch_config : dict
        Architecture configuration
    learn_Z : bool
        Whether Z is learned
        
    Returns:
    --------
    n_params : int
        Total number of parameters
    """
    num_actions = len(arch_config['action_labels'])
    num_states = arch_config['num_states']
    
    # Each profile has: phi (num_actions), xi (num_actions), gamma (1)
    params_per_profile = 2 * num_actions + 1
    profile_params = k * params_per_profile
    
    # Z matrix parameters (if learned)
    if learn_Z:
        # Z is (num_states, k) with softmax constraint per state
        # So (k-1) free parameters per state
        z_params = num_states * (k - 1)
    else:
        z_params = 0
    
    return profile_params + z_params


def run_k_profiles_experiment(k_values=[1, 2, 3, 4, 5], 
                               data_path='data/processed/model_data.pkl',
                               output_dir='results/k_profiles',
                               n_folds=5,
                               n_restarts=3,
                               learn_Z=False,
                               verbose=True):
    """
    Run experiment testing M3 with different numbers of profiles
    
    Parameters:
    -----------
    k_values : list
        List of k values to test
    data_path : str
        Path to processed data
    output_dir : str
        Directory to save results
    n_folds : int
        Number of CV folds
    n_restarts : int
        Number of optimization restarts
    learn_Z : bool
        Whether to learn Z matrix
    verbose : bool
        Print progress
        
    Returns:
    --------
    results : dict
        Complete results for all k values
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if verbose:
        print("="*60)
        print("K-PROFILES EXPERIMENT")
        print("="*60)
        print(f"\nTesting k = {k_values}")
        print(f"CV Folds: {n_folds}")
        print(f"Optimization Restarts: {n_restarts}")
        print(f"Learn Z: {learn_Z}\n")
    
    print(f"Loading data from {data_path}...")
    model_data = load_processed_data(data_path)
    
    # Initialize model components (A, B, D matrices)
    print("Initializing model architecture...")
    A, B, D = initialize_model(ARCHITECTURE_CONFIG, A_MATRIX_CONFIG)
    
    # Extract policies from temporary agent (needed for M3)
    print("Extracting policies from temporary agent...")
    from pymdp import utils
    
    C_dummy = utils.obj_array(ARCHITECTURE_CONFIG['num_modalities'])
    n_obs_surprisal = len(ARCHITECTURE_CONFIG['obs_surprisal_labels'])
    n_obs_length = len(ARCHITECTURE_CONFIG['obs_length_labels'])
    n_obs_switch = len(ARCHITECTURE_CONFIG['obs_switch_labels'])
    
    C_dummy[0] = np.ones(n_obs_surprisal) / n_obs_surprisal
    C_dummy[1] = np.ones(n_obs_length) / n_obs_length
    C_dummy[2] = np.ones(n_obs_switch) / n_obs_switch
    
    temp_agent = Agent(
        A=A, B=B, C=C_dummy, D=D,
        policy_len=ARCHITECTURE_CONFIG['policy_len'],
        inference_horizon=ARCHITECTURE_CONFIG['inference_horizon'],
        control_fac_idx=ARCHITECTURE_CONFIG['control_fac_idx'],
        use_utility=ARCHITECTURE_CONFIG['use_utility'],
        use_states_info_gain=ARCHITECTURE_CONFIG['use_states_info_gain'],
        action_selection=ARCHITECTURE_CONFIG['action_selection'],
        gamma=16.0
    )
    policies = temp_agent.policies
    num_actions = [len(ARCHITECTURE_CONFIG['action_labels'])]
    print(f"Extracted {len(policies)} policies\n")
    
    # Storage for results
    all_results = {
        'k_values': k_values,
        'fold_results': {},  # k -> fold results
        'summary': {},  # k -> summary stats
        'params_count': {},  # k -> n_params
        'training_time': {},  # k -> time in seconds
        'config': {
            'n_folds': n_folds,
            'n_restarts': n_restarts,
            'learn_Z': learn_Z,
            'data_path': data_path
        }
    }
    
    # Test each k value
    for k in k_values:
        print("\n" + "="*60)
        print(f"TESTING k={k} PROFILES")
        print("="*60)
        
        start_time = time.time()
        
        # Create config for this k
        k_config = create_k_profile_config(k, M3_CONFIG, ARCHITECTURE_CONFIG)
        n_params = compute_model_params_count(k, ARCHITECTURE_CONFIG, learn_Z)
        
        if verbose:
            print(f"\nConfiguration:")
            print(f"  Number of profiles: {k}")
            print(f"  Number of parameters: {n_params}")
            print(f"  Initial profiles:")
            for i, profile in enumerate(k_config['initial_profiles']):
                print(f"    Profile {i}: gamma={profile['gamma']:.2f}")
            print(f"\n  Initial Z matrix shape: {k_config['initial_Z'].shape}")
        
        # Create value function using profiles and Z from config
        profiles = k_config['initial_profiles']
        Z = k_config['initial_Z']
        value_fn = make_value_fn_M3(profiles, Z, policies, num_actions, ARCHITECTURE_CONFIG)
        
        # Cross-validation
        print(f"\nRunning {n_folds}-fold cross-validation...")
        
        # Prepare training config for this k
        k_training_config = {
            'n_folds': n_folds,
            'n_restarts': n_restarts,
            'learn_Z': learn_Z,
            'random_seed': 42
        }
        
        cv_results, learned_params = cross_validate_models(
            model_data=model_data,
            value_fns={},  # Empty dict since we're only training M3
            A=A, B=B, D=D,
            policies=policies,
            num_actions=num_actions,
            arch_config=ARCHITECTURE_CONFIG,
            m3_config=k_config,
            training_config=k_training_config,
            n_jobs=1
        )
        
        elapsed_time = time.time() - start_time
        
        # Extract results
        m3_fold_results = cv_results['M3']
        accuracies = [fold['accuracy'] for fold in m3_fold_results]
        log_liks = [fold['total_log_lik'] for fold in m3_fold_results]
        
        # Compute mean and std
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        mean_loglik = np.mean(log_liks)
        std_loglik = np.std(log_liks)
        
        # Compute AIC and BIC on full dataset
        total_loglik = sum(log_liks)
        n_data = len(model_data)
        aic = 2 * n_params - 2 * total_loglik
        bic = n_params * np.log(n_data) - 2 * total_loglik
        
        # Store results
        all_results['fold_results'][k] = {
            'folds': m3_fold_results,
            'learned_params': learned_params
        }
        all_results['params_count'][k] = n_params
        all_results['training_time'][k] = elapsed_time
        all_results['summary'][k] = {
            'mean_accuracy': mean_acc,
            'std_accuracy': std_acc,
            'mean_log_lik': mean_loglik,
            'std_log_lik': std_loglik,
            'total_log_lik': total_loglik,
            'n_params': n_params,
            'aic': aic,
            'bic': bic,
            'training_time': elapsed_time
        }
        
        # Print summary
        print(f"\n" + "="*60)
        print(f"RESULTS FOR k={k}")
        print("="*60)
        print(f"Accuracy:      {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Log-Lik:       {mean_loglik:.2f} ± {std_loglik:.2f}")
        print(f"AIC:           {aic:.2f}")
        print(f"BIC:           {bic:.2f}")
        print(f"Num Params:    {n_params}")
        print(f"Training Time: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)")
    
    # Final summary table
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    print(f"\n{'k':>3} | {'Accuracy':>12} | {'LogLik':>12} | {'AIC':>10} | {'BIC':>10} | {'Params':>7} | {'Time(s)':>8}")
    print("-" * 80)
    
    for k in k_values:
        summary = all_results['summary'][k]
        print(f"{k:>3} | {summary['mean_accuracy']:>5.4f} ± {summary['std_accuracy']:>4.4f} | "
              f"{summary['mean_log_lik']:>6.2f} ± {summary['std_log_lik']:>4.2f} | "
              f"{summary['aic']:>10.2f} | {summary['bic']:>10.2f} | "
              f"{summary['n_params']:>7} | {summary['training_time']:>8.2f}")
    
    # Find best k by each criterion
    print("\n" + "="*60)
    print("BEST k BY CRITERION")
    print("="*60)
    
    best_acc_k = max(k_values, key=lambda k: all_results['summary'][k]['mean_accuracy'])
    best_loglik_k = max(k_values, key=lambda k: all_results['summary'][k]['mean_log_lik'])
    best_aic_k = min(k_values, key=lambda k: all_results['summary'][k]['aic'])
    best_bic_k = min(k_values, key=lambda k: all_results['summary'][k]['bic'])
    
    print(f"Best Accuracy:      k={best_acc_k} ({all_results['summary'][best_acc_k]['mean_accuracy']:.4f})")
    print(f"Best Log-Lik:       k={best_loglik_k} ({all_results['summary'][best_loglik_k]['mean_log_lik']:.2f})")
    print(f"Best AIC:           k={best_aic_k} ({all_results['summary'][best_aic_k]['aic']:.2f})")
    print(f"Best BIC (optimal): k={best_bic_k} ({all_results['summary'][best_bic_k]['bic']:.2f})")
    
    # Save results
    results_file = output_path / 'k_profiles_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"\nResults saved to {results_file}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_k_profiles_results(all_results, output_path)
    
    return all_results


def plot_k_profiles_results(results, output_dir):
    """
    Create visualizations for k-profiles experiment
    
    Parameters:
    -----------
    results : dict
        Results from run_k_profiles_experiment
    output_dir : Path
        Directory to save plots
    """
    k_values = results['k_values']
    summary = results['summary']
    
    # Extract data for plotting
    accuracies = [summary[k]['mean_accuracy'] for k in k_values]
    acc_stds = [summary[k]['std_accuracy'] for k in k_values]
    log_liks = [summary[k]['mean_log_lik'] for k in k_values]
    loglik_stds = [summary[k]['std_log_lik'] for k in k_values]
    aics = [summary[k]['aic'] for k in k_values]
    bics = [summary[k]['bic'] for k in k_values]
    n_params = [summary[k]['n_params'] for k in k_values]
    times = [summary[k]['training_time'] / 60 for k in k_values]  # Convert to minutes
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('K-Profiles Experiment Results', fontsize=16, fontweight='bold')
    
    # 1. Accuracy vs k
    ax = axes[0, 0]
    ax.errorbar(k_values, accuracies, yerr=acc_stds, marker='o', capsize=5, linewidth=2, markersize=8)
    ax.set_xlabel('Number of Profiles (k)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Accuracy vs Number of Profiles', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    # 2. Log-Likelihood vs k
    ax = axes[0, 1]
    ax.errorbar(k_values, log_liks, yerr=loglik_stds, marker='o', capsize=5, linewidth=2, markersize=8, color='green')
    ax.set_xlabel('Number of Profiles (k)', fontsize=11)
    ax.set_ylabel('Mean Log-Likelihood', fontsize=11)
    ax.set_title('Log-Likelihood vs Number of Profiles', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    # 3. AIC vs k (lower is better)
    ax = axes[0, 2]
    ax.plot(k_values, aics, marker='s', linewidth=2, markersize=8, color='red', label='AIC')
    ax.plot(k_values, bics, marker='^', linewidth=2, markersize=8, color='purple', label='BIC')
    ax.set_xlabel('Number of Profiles (k)', fontsize=11)
    ax.set_ylabel('Information Criterion', fontsize=11)
    ax.set_title('Model Selection Criteria (lower = better)', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    # 4. BIC Elbow Plot (key plot!)
    ax = axes[1, 0]
    ax.plot(k_values, bics, marker='o', linewidth=3, markersize=10, color='purple')
    best_k = min(k_values, key=lambda k: summary[k]['bic'])
    best_bic = summary[best_k]['bic']
    ax.scatter([best_k], [best_bic], color='red', s=200, zorder=5, marker='*', 
               label=f'Optimal k={best_k}')
    ax.set_xlabel('Number of Profiles (k)', fontsize=11, fontweight='bold')
    ax.set_ylabel('BIC', fontsize=11, fontweight='bold')
    ax.set_title('BIC Elbow Plot (Optimal k Selection)', fontweight='bold', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    # 5. Number of Parameters vs k
    ax = axes[1, 1]
    ax.plot(k_values, n_params, marker='d', linewidth=2, markersize=8, color='orange')
    ax.set_xlabel('Number of Profiles (k)', fontsize=11)
    ax.set_ylabel('Number of Parameters', fontsize=11)
    ax.set_title('Model Complexity', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(k_values)
    
    # 6. Training Time vs k
    ax = axes[1, 2]
    ax.bar(k_values, times, color='steelblue', alpha=0.7, edgecolor='black')
    ax.set_xlabel('Number of Profiles (k)', fontsize=11)
    ax.set_ylabel('Training Time (minutes)', fontsize=11)
    ax.set_title('Computational Cost', fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(k_values)
    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / 'k_profiles_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {output_file}")
    plt.close()
    
    # Create summary CSV
    summary_df = pd.DataFrame({
        'k': k_values,
        'accuracy_mean': accuracies,
        'accuracy_std': acc_stds,
        'loglik_mean': log_liks,
        'loglik_std': loglik_stds,
        'AIC': aics,
        'BIC': bics,
        'n_params': n_params,
        'training_time_min': times
    })
    
    csv_file = output_dir / 'k_profiles_summary.csv'
    summary_df.to_csv(csv_file, index=False)
    print(f"Saved summary table to {csv_file}")


def main():
    """Main entry point"""
    # Configuration
    k_values = [1, 2, 3, 4, 5]
    data_path = 'data/processed/model_data.pkl'
    output_dir = 'results/k_profiles'
    n_folds = 5
    n_restarts = 3
    learn_Z = False  # Set to True to also optimize Z matrix
    
    # Run experiment
    results = run_k_profiles_experiment(
        k_values=k_values,
        data_path=data_path,
        output_dir=output_dir,
        n_folds=n_folds,
        n_restarts=n_restarts,
        learn_Z=learn_Z,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("K-PROFILES EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("  - k_profiles_results.pkl")
    print("  - k_profiles_comparison.png")
    print("  - k_profiles_summary.csv")


if __name__ == '__main__':
    main()
