"""
Baseline Models Module
Implements logistic regression baselines for comparison with Active Inference models
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss


def train_logistic_baseline(train_data, test_data, features, model_name='LR', 
                            scale_std=0.5, random_state=42):
    """
    Train a logistic regression baseline model
    
    Parameters:
    -----------
    train_data : DataFrame
        Training data
    test_data : DataFrame
        Test data
    features : list of str
        List of feature column names to use
    model_name : str
        Name of the model (for reporting)
    scale_std : float
        Target standard deviation for feature scaling (following Calvillo et al.)
    random_state : int
        Random seed
        
    Returns:
    --------
    results : dict
        Dictionary containing accuracy, log-likelihood, predictions, and model
    """
    # Prepare features and targets
    X_train = train_data[features].values
    X_test = test_data[features].values
    y_train = train_data['cs_binary'].values
    y_test = test_data['cs_binary'].values
    
    # Standardize to mean=0, std=scale_std (following Calvillo et al. 2020)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) * scale_std
    X_test_scaled = scaler.transform(X_test) * scale_std
    
    # Train logistic regression
    lr = LogisticRegression(random_state=random_state, max_iter=1000, solver='lbfgs')
    lr.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Compute log-likelihood manually to match Active Inference calculation
    # log p(y|x) = sum_i [y_i * log(p_i) + (1-y_i) * log(1-p_i)]
    log_likelihoods = y_test * np.log(y_pred_proba + 1e-12) + (1 - y_test) * np.log(1 - y_pred_proba + 1e-12)
    total_log_lik = np.sum(log_likelihoods)
    
    # Calculate AIC and BIC
    n_samples = len(y_test)
    n_params = len(features) + 1  # coefficients + intercept
    aic = 2 * n_params - 2 * total_log_lik
    bic = np.log(n_samples) * n_params - 2 * total_log_lik
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'total_log_lik': total_log_lik,
        'aic': aic,
        'bic': bic,
        'n_params': n_params,
        'predictions': y_pred_proba,
        'actual': y_test,
        'model': lr,
        'scaler': scaler,
        'features': features,
        'log_likelihoods': log_likelihoods  # Add individual log-likelihoods
    }
    
    return results


def cross_validate_logistic_baselines(model_data, kfold_splitter, pair_ids, 
                                      baseline_configs, verbose=True):
    """
    Cross-validate multiple logistic regression baseline models
    
    Parameters:
    -----------
    model_data : DataFrame
        Complete dataset
    kfold_splitter : KFold
        KFold cross-validator
    pair_ids : np.ndarray
        Array of pair IDs
    baseline_configs : dict
        Dictionary mapping baseline names to feature lists
        Example: {'LR1': ['surprisal_first_cs_word_trans'],
                  'LR2': ['surprisal_first_cs_word_trans', 'translation_sentence_length']}
    verbose : bool
        Print progress
        
    Returns:
    --------
    cv_results : dict
        Dictionary mapping baseline names to list of fold results
    """
    if verbose:
        print(f"\n{'='*60}")
        print("LOGISTIC REGRESSION BASELINES")
        print(f"{'='*60}")
        print(f"Models: {list(baseline_configs.keys())}")
    
    cv_results = {name: [] for name in baseline_configs.keys()}
    
    for fold_idx, (train_pair_indices, test_pair_indices) in enumerate(kfold_splitter.split(pair_ids)):
        if verbose:
            print(f"\nFold {fold_idx + 1}/{kfold_splitter.n_splits}")
        
        # Get train and test data
        train_pair_ids = pair_ids[train_pair_indices]
        test_pair_ids = pair_ids[test_pair_indices]
        
        train_data = model_data[model_data['pair_id'].isin(train_pair_ids)].copy()
        test_data = model_data[model_data['pair_id'].isin(test_pair_ids)].copy()
        
        # Train each baseline
        for baseline_name, features in baseline_configs.items():
            results = train_logistic_baseline(
                train_data, test_data, features, 
                model_name=baseline_name
            )
            cv_results[baseline_name].append(results)
            
            if verbose:
                print(f"  {baseline_name} Accuracy: {results['accuracy']:.4f}")
    
    return cv_results


def print_baseline_summary(cv_results, baseline_configs):
    """
    Print summary statistics for baseline models
    
    Parameters:
    -----------
    cv_results : dict
        Dictionary mapping baseline names to fold results
    baseline_configs : dict
        Dictionary mapping baseline names to feature lists
    """
    print("\n" + "="*60)
    print("BASELINE MODEL SUMMARY")
    print("="*60)
    
    for baseline_name in cv_results.keys():
        fold_results = cv_results[baseline_name]
        
        accs = [r['accuracy'] for r in fold_results]
        logliks = [r['total_log_lik'] for r in fold_results]
        aics = [r['aic'] for r in fold_results]
        bics = [r['bic'] for r in fold_results]
        
        print(f"\n{baseline_name}: {baseline_configs[baseline_name]}")
        print(f"  Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"  Log-Lik:   {np.mean(logliks):.2f} ± {np.std(logliks):.2f}")
        print(f"  AIC:       {np.mean(aics):.2f} ± {np.std(aics):.2f}")
        print(f"  BIC:       {np.mean(bics):.2f} ± {np.std(bics):.2f}")
        print(f"  N params:  {fold_results[0]['n_params']}")


def get_baseline_configs():
    """
    Get default baseline configurations
    
    Returns:
    --------
    baseline_configs : dict
        Dictionary mapping baseline names to feature lists
    """
    return {
        'LR1': ['surprisal_first_cs_word_trans'],
        'LR2': ['surprisal_first_cs_word_trans', 'translation_sentence_length']
        # Note: LR3 with frequency removed because frequency column not in processed data
        # To add frequency: include 'frequency_negative_ln_first_cs_word_trans' in 
        # prepare_model_data() in src/data_processing.py
    }
