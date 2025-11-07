"""
Cross-Validation Framework
"""

import numpy as np
from sklearn.model_selection import KFold
from .optimizer import fit_M3_parameters
from ..models.value_functions import make_value_fn_M3
from ..evaluation.metrics import evaluate_model
import time
from joblib import Parallel, delayed


def _process_fold(fold_idx, train_pair_ids, test_pair_ids, model_data, pair_ids,
                  value_fns, A, B, D, policies, num_actions, arch_config, 
                  m3_config, training_config, warm_start_params):
    """
    Process a single fold (for parallel execution)
    
    Parameters:
    -----------
    fold_idx : int
        Fold index
    train_pair_ids : np.ndarray
        Training pair IDs
    test_pair_ids : np.ndarray
        Test pair IDs
    model_data : DataFrame
        Complete dataset
    pair_ids : np.ndarray
        All pair IDs
    value_fns : dict
        Value functions for M1, M2
    A, B, D : object arrays
        Generative model matrices
    policies : list
        Policy list
    num_actions : list
        Number of actions per factor
    arch_config : dict
        Architecture configuration
    m3_config : dict
        M3 configuration
    training_config : dict
        Training configuration
    warm_start_params : dict or None
        Warm start parameters from previous fold
        
    Returns:
    --------
    fold_results : dict
        Results for this fold
    """
    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/{training_config['n_folds']}")
    print(f"{'='*60}")
    
    # Split data by pairs
    train_data = model_data[model_data['pair_id'].isin(train_pair_ids)].copy()
    test_data = model_data[model_data['pair_id'].isin(test_pair_ids)].copy()
    
    print(f"Train: {len(train_data)} sentences ({len(train_pair_ids)} pairs)")
    print(f"Test:  {len(test_data)} sentences ({len(test_pair_ids)} pairs)")
    
    fold_results = {'fold_idx': fold_idx}
    
    # Evaluate M1, M2 on test data
    for model_name, value_fn in value_fns.items():
        results = evaluate_model(
            model_name, value_fn, test_data, A, B, D, arch_config, verbose=False
        )
        fold_results[model_name] = results
        print(f"{model_name} Accuracy: {results['accuracy']:.4f}")
    
    # Train M3 on training data
    print(f"\nTraining M3 on fold {fold_idx + 1}...")
    fold_start = time.time()
    learned_params_fold = fit_M3_parameters(
        train_data, A, D, m3_config['initial_Z'], policies, num_actions,
        arch_config, learn_Z=training_config['learn_Z'],
        n_restarts=training_config['n_restarts'], verbose=False,
        warm_start_params=warm_start_params
    )
    fold_time = time.time() - fold_start
    print(f"M3 training completed in {fold_time:.2f}s")
    
    # Evaluate M3 on test data
    value_fn_M3_fold = make_value_fn_M3(
        learned_params_fold['profiles'],
        learned_params_fold['Z'],
        policies, num_actions, arch_config
    )
    results_M3 = evaluate_model(
        'M3', value_fn_M3_fold, test_data, A, B, D, arch_config, verbose=False
    )
    fold_results['M3'] = results_M3
    fold_results['M3_params'] = learned_params_fold
    print(f"M3 Accuracy: {results_M3['accuracy']:.4f}")
    
    return fold_results


def cross_validate_models(model_data, value_fns, A, B, D, policies, num_actions, 
                          arch_config, m3_config, training_config, n_jobs=1):
    """
    Perform cross-validation for all models
    
    Parameters:
    -----------
    model_data : DataFrame
        Complete dataset
    value_fns : dict
        Dictionary mapping model names to value functions (for M1, M2)
    A, B, D : object arrays
        Generative model matrices
    policies : list
        Policy list
    num_actions : list
        Number of actions per factor
    arch_config : dict
        Architecture configuration
    m3_config : dict
        M3 configuration
    training_config : dict
        Training configuration
    n_jobs : int
        Number of parallel jobs (1 = sequential, -1 = all cores)
        
    Returns:
    --------
    cv_results : dict
        Dictionary mapping model names to list of fold results
    learned_params : list
        List of learned M3 parameters for each fold
    """
    n_folds = training_config['n_folds']
    pair_ids = model_data['pair_id'].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, 
               random_state=training_config.get('random_seed', 42))
    
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION ({n_folds} folds)")
    print(f"{'='*60}")
    print(f"Splitting by pairs to keep matched sentences together")
    print(f"Total pairs: {len(pair_ids)}")
    if n_jobs != 1:
        print(f"Parallel execution: {n_jobs} jobs")
    
    cv_start_time = time.time()
    
    # Prepare fold splits
    fold_splits = list(kf.split(pair_ids))
    
    # Sequential processing with warm start
    if n_jobs == 1:
        cv_results = {name: [] for name in value_fns.keys()}
        cv_results['M3'] = []
        learned_params = []
        warm_start_params = None
        
        for fold_idx, (train_pair_indices, test_pair_indices) in enumerate(fold_splits):
            train_pair_ids = pair_ids[train_pair_indices]
            test_pair_ids = pair_ids[test_pair_indices]
            
            fold_results = _process_fold(
                fold_idx, train_pair_ids, test_pair_ids, model_data, pair_ids,
                value_fns, A, B, D, policies, num_actions, arch_config,
                m3_config, training_config, warm_start_params
            )
            
            # Store results
            for model_name in value_fns.keys():
                cv_results[model_name].append(fold_results[model_name])
            cv_results['M3'].append(fold_results['M3'])
            learned_params.append(fold_results['M3_params'])
            
            # Use this fold's params for warm start in next fold
            warm_start_params = fold_results['M3_params']
    
    # Parallel processing (no warm start across folds)
    else:
        print("\nNote: Parallel execution disables warm-start between folds")
        
        fold_results_list = Parallel(n_jobs=n_jobs)(
            delayed(_process_fold)(
                fold_idx,
                pair_ids[train_pair_indices],
                pair_ids[test_pair_indices],
                model_data, pair_ids, value_fns, A, B, D, policies,
                num_actions, arch_config, m3_config, training_config, None
            )
            for fold_idx, (train_pair_indices, test_pair_indices) in enumerate(fold_splits)
        )
        
        # Organize results
        cv_results = {name: [] for name in value_fns.keys()}
        cv_results['M3'] = []
        learned_params = []
        
        for fold_results in fold_results_list:
            for model_name in value_fns.keys():
                cv_results[model_name].append(fold_results[model_name])
            cv_results['M3'].append(fold_results['M3'])
            learned_params.append(fold_results['M3_params'])
    
    cv_time = time.time() - cv_start_time
    print(f"\n{'='*60}")
    print(f"CROSS-VALIDATION COMPLETED in {cv_time:.2f}s ({cv_time/60:.2f} min)")
    print(f"{'='*60}")
    
    return cv_results, learned_params


def print_cv_results(cv_results):
    """
    Print cross-validation results summary
    
    Parameters:
    -----------
    cv_results : dict
        Dictionary mapping model names to list of fold results
    """
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    
    print(f"\n{'Model':<20s} {'Mean Accuracy':>15s} {'Mean LogLik':>12s}")
    print("-" * 60)
    
    for model_name, fold_results in cv_results.items():
        accs = [r['accuracy'] for r in fold_results]
        logliks = [r['total_log_lik'] for r in fold_results]
        
        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_loglik = np.mean(logliks)
        std_loglik = np.std(logliks)
        
        print(f"{model_name:<20s} "
              f"{mean_acc:>7.4f} ± {std_acc:<5.4f} "
              f"{mean_loglik:>7.2f} ± {std_loglik:<4.2f}")
    
    return {
        model: {
            'mean_accuracy': np.mean([r['accuracy'] for r in results]),
            'std_accuracy': np.std([r['accuracy'] for r in results]),
            'mean_loglik': np.mean([r['total_log_lik'] for r in results]),
            'std_loglik': np.std([r['total_log_lik'] for r in results])
        }
        for model, results in cv_results.items()
    }
