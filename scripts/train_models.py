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
                                   print_baseline_summary, get_baseline_configs)
from src.training.cross_validation import cross_validate_models, print_cv_results
from src.evaluation.metrics import evaluate_model, compare_models
from src.evaluation.visualization import plot_model_comparison, plot_m3_mechanism
from pymdp.agent import Agent
from pymdp import control
import numpy as np


def main():
    """Main training pipeline"""
    
    print("="*60)
    print("ACTIVE INFERENCE CODE-SWITCHING MODEL TRAINING")
    print("="*60)
    
    # ============================================================
    # STEP 1: DATA PREPROCESSING
    # ============================================================
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
    
    A, B, D = initialize_model(ARCHITECTURE_CONFIG, A_MATRIX_CONFIG)
    
    # ============================================================
    # STEP 3: CREATE VALUE FUNCTIONS
    # ============================================================
    print("\n### STEP 3: CREATE VALUE FUNCTIONS ###\n")
    
    # M1: Static precision
    value_fn_M1 = make_value_fn_M1(M1_CONFIG, ARCHITECTURE_CONFIG)
    print(f"✓ Created {M1_CONFIG['name']}: {M1_CONFIG['description']}")
    
    # M2: Entropy-coupled precision
    value_fn_M2 = make_value_fn_M2(M2_CONFIG, ARCHITECTURE_CONFIG)
    print(f"✓ Created {M2_CONFIG['name']}: {M2_CONFIG['description']}")
    
    # Extract policies for M3
    print("\nExtracting policies from temporary agent...")
    C_dummy = utils.obj_array(ARCHITECTURE_CONFIG['num_modalities'])
    
    # Create C matrices with correct dimensions based on actual observation space
    n_obs_surprisal = len(ARCHITECTURE_CONFIG['obs_surprisal_labels'])
    n_obs_length = len(ARCHITECTURE_CONFIG['obs_length_labels'])
    n_obs_switch = len(ARCHITECTURE_CONFIG['obs_switch_labels'])
    
    C_dummy[0] = np.ones(n_obs_surprisal) / n_obs_surprisal  # Uniform over surprisal obs
    C_dummy[1] = np.ones(n_obs_length) / n_obs_length        # Uniform over length obs
    C_dummy[2] = np.ones(n_obs_switch) / n_obs_switch        # Uniform over switch obs
    
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
    print(f"Extracted {len(policies)} policies")
    
    # M3: Profile-based precision
    profile_manager = create_profile_manager(M3_CONFIG)
    value_fn_M3 = make_value_fn_M3(
        profile_manager.get_profiles(),
        profile_manager.get_Z(),
        policies, num_actions,
        ARCHITECTURE_CONFIG
    )
    print(f"✓ Created {M3_CONFIG['name']}: {M3_CONFIG['description']}")
    
    # Test value functions
    test_value_functions(value_fn_M1, value_fn_M2, value_fn_M3)
    
    # ============================================================
    # STEP 4: EVALUATE ON FULL DATASET (Initial)
    # ============================================================
    print("\n### STEP 4: INITIAL EVALUATION ON FULL DATASET ###\n")
    
    results_M1 = evaluate_model('M1', value_fn_M1, model_data, A, B, D, 
                                ARCHITECTURE_CONFIG, M1_CONFIG, verbose=True)
    results_M2 = evaluate_model('M2', value_fn_M2, model_data, A, B, D,
                                ARCHITECTURE_CONFIG, M2_CONFIG, verbose=True)
    results_M3_initial = evaluate_model('M3_initial', value_fn_M3, model_data, A, B, D,
                                        ARCHITECTURE_CONFIG, M3_CONFIG, verbose=True)
    
    # ============================================================
    # STEP 5: CROSS-VALIDATION WITH LEARNING
    # ============================================================
    print("\n### STEP 5: CROSS-VALIDATION WITH M3 LEARNING ###\n")
    
    value_fns = {'M1': value_fn_M1, 'M2': value_fn_M2}
    
    cv_results, learned_params = cross_validate_models(
        model_data, value_fns, A, B, D, policies, num_actions,
        ARCHITECTURE_CONFIG, M3_CONFIG, TRAINING_CONFIG,
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
    # STEP 6: FINAL EVALUATION WITH BEST LEARNED M3
    # ============================================================
    print("\n### STEP 6: FINAL EVALUATION WITH LEARNED M3 ###\n")
    
    # Use parameters from best fold
    best_fold_idx = np.argmax([r['accuracy'] for r in cv_results['M3']])
    best_params = learned_params[best_fold_idx]
    print(f"Using parameters from best fold ({best_fold_idx + 1})")
    
    # Create value function with learned parameters
    value_fn_M3_learned = make_value_fn_M3(
        best_params['profiles'],
        best_params['Z'],
        policies, num_actions,
        ARCHITECTURE_CONFIG
    )
    
    # Evaluate on full dataset
    results_M3_learned = evaluate_model('M3_learned', value_fn_M3_learned, model_data,
                                       A, B, D, ARCHITECTURE_CONFIG, M3_CONFIG, verbose=True)
    
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
    
    # Add baseline results (using mean across folds)
    for baseline_name in cv_results_baselines.keys():
        baseline_folds = cv_results_baselines[baseline_name]
        # Concatenate all predictions and actuals across folds
        all_predictions = np.concatenate([r['predictions'] for r in baseline_folds])
        all_actuals = np.concatenate([r['actual'] for r in baseline_folds])
        all_log_liks = np.concatenate([r['log_likelihoods'] for r in baseline_folds])
        
        # Total log-likelihood is the SUM across all samples (not mean)
        total_log_lik = np.sum(all_log_liks)
        
        # Re-compute AIC and BIC on full dataset
        n_samples = len(all_actuals)
        n_params = baseline_folds[0]['n_params']
        aic = 2 * n_params - 2 * total_log_lik
        bic = np.log(n_samples) * n_params - 2 * total_log_lik
        
        # Aggregate results from all folds
        all_results[baseline_name] = {
            'accuracy': np.mean([r['accuracy'] for r in baseline_folds]),
            'total_log_lik': total_log_lik,  # Fixed: SUM not mean
            'aic': aic,  # Re-computed on full dataset
            'bic': bic,  # Re-computed on full dataset
            'n_params': n_params,
            'predictions': all_predictions,
            'actual': all_actuals,
            'log_likelihoods': all_log_liks,
            'gammas': np.array([])  # Empty for baselines
        }
    
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
        'best_params': best_params,
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
    
    return all_results, cv_results, best_params


if __name__ == '__main__':
    # Import additional required modules
    from pymdp import utils
    
    main()
