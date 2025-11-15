"""
Model Evaluation Module
Handles inference and evaluation of Active Inference models
"""

import numpy as np
from pymdp import utils
from pymdp.agent import Agent
from ..models.active_inference import softmax


def evaluate_model(model_name, value_fn, data, A, B, D, arch_config, model_config=None, verbose=False):
    """
    Evaluate a model on the code-switching data
    
    For each sentence:
    1. Observe surprisal and length
    2. Infer cognitive state beliefs
    3. Use value function to get C, E, gamma
    4. Compute action probabilities
    5. Compare to actual code-switching outcome
    6. Accumulate log-likelihood
    
    Parameters:
    -----------
    model_name : str
        Name of model (M1, M2, M3)
    value_fn : function
        Value function that returns (C, E, gamma) given beliefs
    data : DataFrame
        Code-switching dataset
    A, B, D : object arrays
        Generative model matrices
    arch_config : dict
        Architecture configuration
    model_config : dict, optional
        Model-specific configuration (for computing n_params)
    verbose : bool
        Print progress
        
    Returns:
    --------
    results : dict
        Contains log_likelihoods, accuracy, inferred_states, etc.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name}")
        print(f"{'='*60}")
    
    n_samples = len(data)
    log_likelihoods = []
    inferred_states = []
    gammas = []
    predictions = []
    
    # Process each sentence
    for idx in range(n_samples):
        if verbose and idx % 500 == 0:
            print(f"Processing sentence {idx}/{n_samples}...")
        
        # Get observations for this sentence
        obs_surprisal_idx = int(data.iloc[idx]['surprisal_idx'])
        obs_length_idx = int(data.iloc[idx]['length_idx'])
        actual_switch = int(data.iloc[idx]['cs_binary'])
        
        # State inference using only surprisal and length
        # Compute posterior: p(s|o_surprisal, o_length) ∝ p(o_surprisal|s) * p(o_length|s) * p(s)
        likelihood_surprisal = A[0][obs_surprisal_idx, :]
        likelihood_length = A[1][obs_length_idx, :]
        prior = D[0]
        
        posterior = likelihood_surprisal * likelihood_length * prior
        posterior = posterior / (posterior.sum() + 1e-12)
        
        # Get value function parameters based on current beliefs
        C_t, E_t, gamma_t = value_fn(posterior, idx)
        
        # Compute action probabilities
        if E_t is not None:
            policy_logits = np.log(E_t + 1e-12) + gamma_t * np.log(C_t[2] + 1e-12)
        else:
            policy_logits = gamma_t * np.log(C_t[2] + 1e-12)
        
        action_probs = softmax(policy_logits)
        
        # Probability of the actual observed action
        prob_actual_action = action_probs[actual_switch]
        log_lik = np.log(prob_actual_action + 1e-12)
        
        # Store results
        log_likelihoods.append(log_lik)
        inferred_states.append(posterior)
        gammas.append(gamma_t)
        predictions.append(action_probs[1])  # Probability of switch
    
    # Compute summary statistics
    total_log_lik = np.sum(log_likelihoods)
    mean_log_lik = np.mean(log_likelihoods)
    
    # Accuracy: predict switch if prob > 0.5
    predicted_switches = np.array(predictions) > 0.5
    actual_switches = data['cs_binary'].values
    accuracy = np.mean(predicted_switches == actual_switches)
    
    # Compute number of parameters dynamically
    if model_name == 'M1' or model_name == 'M1_static':
        n_params = 1  # Single gamma
    elif model_name == 'M2' or model_name == 'M2_coupled':
        n_params = 2  # Coupling strength + gamma_base
    elif 'M3' in model_name:
        # M3: num_profiles * 3 free params per profile + (optionally) Z matrix free params
        # Per profile: phi (1 free param - 2nd determined by sum to 1),
        #              xi (1 free param), gamma (1 free param) = 3 total
        # Z matrix: IF learned, num_states * (num_profiles-1) free params
        if model_config is not None:
            num_profiles = model_config.get('num_profiles', 2)
            num_states = arch_config.get('num_states', 2)
            # Handle both int and list types for num_states
            if isinstance(num_states, list):
                num_states = num_states[0]
            
            # Profile parameters only (default: Z is fixed, not learned)
            # For 2 profiles: 2*3 = 6 parameters
            n_params = num_profiles * 3
        else:
            # Fallback: Assume 2 profiles, Z fixed
            n_params = 2 * 3  # 6 parameters
    else:
        n_params = 1  # Default fallback
    
    # AIC and BIC
    aic = 2 * n_params - 2 * total_log_lik
    bic = n_params * np.log(n_samples) - 2 * total_log_lik
    
    results = {
        'model_name': model_name,
        'log_likelihoods': np.array(log_likelihoods),
        'total_log_lik': total_log_lik,
        'mean_log_lik': mean_log_lik,
        'accuracy': accuracy,
        'aic': aic,
        'bic': bic,
        'n_params': n_params,
        'inferred_states': np.array(inferred_states),
        'gammas': np.array(gammas),
        'predictions': np.array(predictions)
    }
    
    if verbose:
        print(f"\nResults for {model_name}:")
        print(f"  Total Log-Likelihood: {total_log_lik:.2f}")
        print(f"  Mean Log-Likelihood:  {mean_log_lik:.4f}")
        print(f"  Accuracy:             {accuracy:.4f}")
        print(f"  AIC:                  {aic:.2f}")
        print(f"  BIC:                  {bic:.2f}")
        print(f"  Mean Gamma:           {np.mean(gammas):.4f}")
        print(f"  Std Gamma:            {np.std(gammas):.4f}")
    
    return results


def compare_models(results_dict):
    """
    Compare multiple models and print summary table
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary mapping model names to results dictionaries
    """
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    # Create header
    print(f"\n{'Model':<20s} {'Accuracy':>10s} {'LogLik':>10s} {'AIC':>10s} {'BIC':>10s}")
    print("-" * 62)
    
    # Print each model
    for model_name, results in results_dict.items():
        print(f"{model_name:<20s} "
              f"{results['accuracy']:>10.4f} "
              f"{results['total_log_lik']:>10.2f} "
              f"{results['aic']:>10.2f} "
              f"{results['bic']:>10.2f}")
    
    # Find best model by each criterion
    best_accuracy = max(results_dict.items(), key=lambda x: x[1]['accuracy'])
    best_loglik = max(results_dict.items(), key=lambda x: x[1]['total_log_lik'])
    best_aic = min(results_dict.items(), key=lambda x: x[1]['aic'])
    best_bic = min(results_dict.items(), key=lambda x: x[1]['bic'])
    
    print("\n" + "="*60)
    print("BEST MODELS BY CRITERION")
    print("="*60)
    print(f"Accuracy:      {best_accuracy[0]} ({best_accuracy[1]['accuracy']:.4f})")
    print(f"Log-Likelihood: {best_loglik[0]} ({best_loglik[1]['total_log_lik']:.2f})")
    print(f"AIC:           {best_aic[0]} ({best_aic[1]['aic']:.2f})")
    print(f"BIC:           {best_bic[0]} ({best_bic[1]['bic']:.2f})")
    
    return {
        'best_accuracy': best_accuracy[0],
        'best_loglik': best_loglik[0],
        'best_aic': best_aic[0],
        'best_bic': best_bic[0]
    }
