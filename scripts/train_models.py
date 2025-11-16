"""
Main Training Script
Trains and evaluates all Active Inference models
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import *
from src.data_processing import full_preprocessing_pipeline, load_processed_data
from src.models.active_inference import initialize_model, softmax
from src.models.value_functions import (make_value_fn_M1, make_value_fn_M2, 
                                        make_value_fn_M3, test_value_functions)
from src.models.profiles import create_profile_manager
from src.models.baselines import (cross_validate_logistic_baselines, 
                                   print_baseline_summary, get_baseline_configs,
                                   train_logistic_baseline)
from src.training.cross_validation import cross_validate_models, print_cv_results
from src.evaluation.metrics import evaluate_model, compare_models
from src.evaluation.visualization import plot_model_comparison, plot_m3_mechanism
from pymdp.agent import Agent
from pymdp import control, utils
import numpy as np
import copy
import random


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("ACTIVE INFERENCE CODE-SWITCHING MODEL TRAINING")
    print("="*60)
    
    # ============================================================
    # STEP 1: DATA PREPROCESSING
    # ============================================================

    # Set global random seed for reproducibility (numpy and python random)
    seed = DATA_CONFIG.get('random_seed', 42)
    np.random.seed(seed)
    random.seed(seed)
    print("\n### STEP 1: DATA PREPROCESSING ###\n")
    
    # Check if processed data exists
    processed_path = DATA_CONFIG['processed_data_path']
    if Path(processed_path).exists():
        print(f"Loading preprocessed data from {processed_path}...")
        model_data = load_processed_data(processed_path)
        pairs = None
        stats_dict = None
    else:
        print("Preprocessing raw data...")
        raw_path = DATA_CONFIG['raw_data_path']
        
        if not Path(raw_path).exists():
            print(f"\nERROR: Raw data file not found at {raw_path}")
            print("Please place your CSV file in the data/raw/ directory")
            print("Or update the path in config/model_config.py")
            return
        
        model_data, pairs, stats_dict = full_preprocessing_pipeline(raw_path, DATA_CONFIG)
        
        # Save processed data
        from src.data_processing import save_processed_data
        save_processed_data(model_data, processed_path)
    
    # ============================================================
    # STEP 2: INITIALIZE GENERATIVE MODEL
    # ============================================================
    print("\n### STEP 2: INITIALIZE GENERATIVE MODEL ###\n")

    # Build runtime architecture and A-matrix based on actual discretized bins
    # This prevents mismatches when pd.qcut drops duplicate quantiles
    if 'surprisal_idx' in model_data.columns:
        actual_surprisal_bins = int(model_data['surprisal_idx'].nunique())
    else:
        actual_surprisal_bins = DATA_CONFIG.get('n_bins_surprisal')

    if 'length_idx' in model_data.columns:
        actual_length_bins = int(model_data['length_idx'].nunique())
    else:
        actual_length_bins = DATA_CONFIG.get('n_bins_length')

    runtime_arch = copy.deepcopy(ARCHITECTURE_CONFIG)
    runtime_arch['obs_surprisal_labels'] = [f'surp_bin_{i}' for i in range(actual_surprisal_bins)]
    runtime_arch['obs_length_labels'] = [f'len_bin_{i}' for i in range(actual_length_bins)]

    # Generate corresponding A matrix configuration using the same linear gradient rule
    def _make_a_config(n_surp, n_len):
        surprisal_matrix = np.zeros((n_surp, runtime_arch['num_states']))
        for i in range(n_surp):
            surprisal_matrix[i, 0] = (n_surp - i) / float(n_surp)
            surprisal_matrix[i, 1] = (i + 1) / float(n_surp)
        col_sums = surprisal_matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        surprisal_matrix = surprisal_matrix / col_sums[None, :]

        length_matrix = np.zeros((n_len, runtime_arch['num_states']))
        for i in range(n_len):
            length_matrix[i, 0] = (n_len - i) / float(n_len)
            length_matrix[i, 1] = (i + 1) / float(n_len)
        col_sums_len = length_matrix.sum(axis=0)
        col_sums_len[col_sums_len == 0] = 1.0
        length_matrix = length_matrix / col_sums_len[None, :]

        return {'surprisal': surprisal_matrix, 'length': length_matrix, 'switch': 'uniform'}

    runtime_a_config = _make_a_config(actual_surprisal_bins, actual_length_bins)

    # Use a local runtime architecture so we don't shadow module-level names.
    arch_cfg = runtime_arch
    a_cfg = runtime_a_config

    A, B, D = initialize_model(arch_cfg, a_cfg)
    
    # ============================================================
    # STEP 3: CREATE VALUE FUNCTIONS
    # ============================================================
    print("\n### STEP 3: CREATE VALUE FUNCTIONS ###\n")
    
    # M1: Static precision
    value_fn_M1 = make_value_fn_M1(M1_CONFIG, arch_cfg)
    print(f"✓ Created {M1_CONFIG['name']}: {M1_CONFIG['description']}")
    
    # M2: Entropy-coupled precision
    value_fn_M2 = make_value_fn_M2(M2_CONFIG, arch_cfg)
    print(f"✓ Created {M2_CONFIG['name']}: {M2_CONFIG['description']}")
    
    # Extract policies for M3
    print("\nExtracting policies from temporary agent...")
    C_dummy = utils.obj_array(arch_cfg['num_modalities'])
    
    # Create C matrices with correct dimensions based on actual observation space
    n_obs_surprisal = len(arch_cfg['obs_surprisal_labels'])
    n_obs_length = len(arch_cfg['obs_length_labels'])
    n_obs_switch = len(arch_cfg['obs_switch_labels'])
    
    C_dummy[0] = np.ones(n_obs_surprisal) / n_obs_surprisal  # Uniform over surprisal obs
    C_dummy[1] = np.ones(n_obs_length) / n_obs_length        # Uniform over length obs
    C_dummy[2] = np.ones(n_obs_switch) / n_obs_switch        # Uniform over switch obs
    
    temp_agent = Agent(
        A=A, B=B, C=C_dummy, D=D,
        policy_len=arch_cfg['policy_len'],
        inference_horizon=arch_cfg['inference_horizon'],
        control_fac_idx=arch_cfg['control_fac_idx'],
        use_utility=arch_cfg['use_utility'],
        use_states_info_gain=arch_cfg['use_states_info_gain'],
        action_selection=arch_cfg['action_selection'],
        gamma=arch_cfg.get('policy_extraction_gamma', 16.0)
    )
    policies = temp_agent.policies
    num_actions = [len(arch_cfg['action_labels'])]
    print(f"Extracted {len(policies)} policies")
    
    # M3: Profile-based precision
    profile_manager = create_profile_manager(M3_CONFIG)
    value_fn_M3 = make_value_fn_M3(
        profile_manager.get_profiles(),
        profile_manager.get_Z(),
        policies, num_actions,
        arch_cfg
    )
    print(f"✓ Created {M3_CONFIG['name']}: {M3_CONFIG['description']}")
    
    # Test value functions
    test_value_functions(value_fn_M1, value_fn_M2, value_fn_M3)
    
    # ============================================================
    # STEP 4: EVALUATE ON FULL DATASET (Initial)
    # ============================================================
    print("\n### STEP 4: INITIAL EVALUATION ON FULL DATASET ###\n")
    
    results_M1 = evaluate_model('M1', value_fn_M1, model_data, A, B, D, 
                                arch_cfg, M1_CONFIG, verbose=True)
    results_M2 = evaluate_model('M2', value_fn_M2, model_data, A, B, D,
                                arch_cfg, M2_CONFIG, verbose=True)
    results_M3_initial = evaluate_model('M3_initial', value_fn_M3, model_data, A, B, D,
                                        arch_cfg, M3_CONFIG, verbose=True)
    
    # ============================================================
    # STEP 5: CROSS-VALIDATION WITH LEARNING
    # ============================================================
    print("\n### STEP 5: CROSS-VALIDATION WITH M3 LEARNING ###\n")
    
    value_fns = {'M1': value_fn_M1, 'M2': value_fn_M2}
    
    cv_results, learned_params = cross_validate_models(
        model_data, value_fns, A, B, D, policies, num_actions,
        arch_cfg, M3_CONFIG, TRAINING_CONFIG,
        n_jobs=TRAINING_CONFIG.get('n_jobs', 1)
    )
    
    # Print CV results
    cv_summary = print_cv_results(cv_results)
    
    # ============================================================
    # STEP 5.5: LOGISTIC REGRESSION BASELINES
    # ============================================================
    print("\n### STEP 5.5: LOGISTIC REGRESSION BASELINES ###\n")
    
    # Use same CV splits as Active Inference models
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=TRAINING_CONFIG['n_folds'], shuffle=True, 
               random_state=DATA_CONFIG.get('random_seed', 42))
    
    baseline_configs = get_baseline_configs()
    cv_results_baselines = cross_validate_logistic_baselines(
        model_data, kf, model_data['pair_id'].unique(), baseline_configs, verbose=True
    )
    
    # Print baseline summary
    print_baseline_summary(cv_results_baselines, baseline_configs)
    
    # Add baselines to cv_results for comparison
    cv_results.update(cv_results_baselines)
    
    # ============================================================
    # STEP 6: AGGREGATE M3 PREDICTIONS FROM CROSS-VALIDATION
    # ============================================================
    print("\n### STEP 6: AGGREGATE M3 PREDICTIONS FROM CV FOLDS ###\n")
    
    # The proper way to evaluate M3: use predictions from CV where each data point
    # was in the test set exactly once, so no data leakage
    print("Aggregating M3 predictions from cross-validation folds...")
    
    # Reconstruct full dataset predictions from CV folds
    # Since CV uses KFold on pairs, we need to reconstruct in order
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=TRAINING_CONFIG['n_folds'], shuffle=True, 
               random_state=DATA_CONFIG.get('random_seed', 42))
    pair_ids = model_data['pair_id'].unique()
    
    # Create arrays to store results in original data order
    n_samples = len(model_data)
    m3_predictions_full = np.zeros(n_samples)
    m3_logliks_full = np.zeros(n_samples)
    
    # Fill in predictions from each fold
    for fold_idx, (train_pair_indices, test_pair_indices) in enumerate(kf.split(pair_ids)):
        test_pair_ids = pair_ids[test_pair_indices]
        test_indices = model_data[model_data['pair_id'].isin(test_pair_ids)].index
        
        # Get predictions from this fold
        fold_predictions = cv_results['M3'][fold_idx]['predictions']
        fold_logliks = cv_results['M3'][fold_idx]['log_likelihoods']
        
        # Place them in the correct positions
        m3_predictions_full[test_indices] = fold_predictions
        m3_logliks_full[test_indices] = fold_logliks
    
    # Get actuals from original data
    m3_actuals_full = model_data['cs_binary'].values
    
    # Compute aggregate metrics
    m3_accuracy = np.mean((m3_predictions_full > 0.5) == m3_actuals_full)
    m3_total_loglik = np.sum(m3_logliks_full)
    n_params = M3_CONFIG['num_profiles'] * 3  # phi, xi, gamma per profile
    m3_aic = 2 * n_params - 2 * m3_total_loglik
    m3_bic = np.log(n_samples) * n_params - 2 * m3_total_loglik
    
    results_M3_learned = {
        'model_name': 'M3_learned',
        'accuracy': m3_accuracy,
        'total_log_lik': m3_total_loglik,
        'aic': m3_aic,
        'bic': m3_bic,
        'n_params': n_params,
        'predictions': m3_predictions_full,
        'actual': m3_actuals_full,
        'log_likelihoods': m3_logliks_full,
        'gammas': np.array([]),  # Not storing gammas from CV
        'inferred_states': np.array([])  # Not storing states from CV
    }
    
    print(f"M3 Aggregated Results:")
    print(f"  Accuracy: {m3_accuracy:.4f}")
    print(f"  Total Log-Likelihood: {m3_total_loglik:.2f}")
    print(f"  AIC: {m3_aic:.2f}")
    print(f"  BIC: {m3_bic:.2f}")
    print("\nNote: These are true out-of-sample predictions (no data leakage).")
    
    # ============================================================
    # STEP 7: COMPARE ALL MODELS
    # ============================================================
    print("\n### STEP 7: FINAL MODEL COMPARISON ###\n")
    
    all_results = {
        'M1': results_M1,
        'M2': results_M2,
        'M3_initial': results_M3_initial,
        'M3_learned': results_M3_learned
    }
    
    # Train baselines on full dataset for fair comparison
    print("\nTraining baseline models on full dataset for final comparison...")
    baseline_configs = get_baseline_configs()
    
    for baseline_name, features in baseline_configs.items():
        # Train on full dataset (for final comparison, not CV)
        # Note: This is philosophically different from CV - here we're asking
        # "how well does each model fit the full dataset" not "how well does it generalize"
        lr_result = train_logistic_baseline(
            model_data, model_data, features,  # Train and test on same data
            model_name=baseline_name
        )
        
        all_results[baseline_name] = {
            'model_name': baseline_name,
            'accuracy': lr_result['accuracy'],
            'total_log_lik': lr_result['total_log_lik'],
            'aic': lr_result['aic'],
            'bic': lr_result['bic'],
            'n_params': lr_result['n_params'],
            'predictions': lr_result['predictions'],
            'actual': lr_result['actual'],
            'log_likelihoods': lr_result['log_likelihoods'],
            'gammas': np.array([])  # Empty for baselines
        }
        print(f"  {baseline_name}: Accuracy={lr_result['accuracy']:.4f}, LogLik={lr_result['total_log_lik']:.2f}")
    
    print("\nNote: All models evaluated on full dataset.")
    print("CV results above show generalization performance.")
    print("Final comparison shows in-sample fit (for model selection with AIC/BIC).")
    
    best_models = compare_models(all_results)
    
    # ============================================================
    # STEP 8: VISUALIZATIONS
    # ============================================================
    print("\n### STEP 8: GENERATING VISUALIZATIONS ###\n")
    
    results_dir = VIZ_CONFIG['results_dir']
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Model comparison plot
    plot_model_comparison(all_results, VIZ_CONFIG, 
                         save_path=f'{results_dir}/model_comparison.png')
    
    # M3 mechanism plot
    plot_m3_mechanism(results_M3_learned, model_data, VIZ_CONFIG,
                     save_path=f'{results_dir}/m3_mechanism.png')
    
    # ============================================================
    # STEP 9: SAVE RESULTS
    # ============================================================
    print("\n### STEP 9: SAVING RESULTS ###\n")
    
    import pickle
    
    results_to_save = {
        'all_results': all_results,
        'cv_results': cv_results,
        'cv_results_baselines': cv_results_baselines,
        'cv_summary': cv_summary,
        'learned_params': learned_params,
        'best_models': best_models,
        'baseline_configs': baseline_configs
    }
    
    results_file = f'{results_dir}/training_results.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(results_to_save, f)
    print(f"Saved results to {results_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"\nBest Model by Accuracy: {best_models['best_accuracy']}")
    print(f"Best Model by Log-Likelihood: {best_models['best_loglik']}")
    print(f"\nResults saved to: {results_dir}/")
    print(f"  - model_comparison.png")
    print(f"  - m3_mechanism.png")
    print(f"  - training_results.pkl")
    
    return all_results, cv_results, learned_params


if __name__ == '__main__':
    # Import additional required modules
    from pymdp import utils
    main()
