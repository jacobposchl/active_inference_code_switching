"""
Model Training and Optimization Module
Handles parameter learning for M3 model
"""

import numpy as np
from scipy.optimize import minimize
from sklearn.model_selection import KFold
from pymdp import utils
from ..models.active_inference import softmax
from ..models.value_functions import make_value_fn_M3
from ..evaluation.metrics import evaluate_model
import time
from functools import lru_cache


def precompute_posteriors(data, A, D):
    """
    Pre-compute posterior state distributions for all data points
    This avoids redundant computation during optimization
    
    Parameters:
    -----------
    data : DataFrame
        Data with observation indices
    A : object array
        Observation model
    D : object array
        Prior
        
    Returns:
    --------
    posteriors : np.ndarray
        Array of posterior distributions (n_data, n_states)
    """
    n_data = len(data)
    n_states = len(D[0])
    posteriors = np.zeros((n_data, n_states))
    
    for idx in range(n_data):
        obs_surprisal_idx = int(data.iloc[idx]['surprisal_idx'])
        obs_length_idx = int(data.iloc[idx]['length_idx'])
        
        likelihood_surprisal = A[0][obs_surprisal_idx, :]
        likelihood_length = A[1][obs_length_idx, :]
        prior = D[0]
        
        posterior = likelihood_surprisal * likelihood_length * prior
        posteriors[idx] = posterior / (posterior.sum() + 1e-12)
    
    return posteriors


def compute_log_likelihood_M3(params_flat, data, A, D, policies, num_actions, 
                               arch_config, Z_fixed=None, learn_Z=False, posteriors=None):
    """
    Compute negative log-likelihood for M3 given parameters
    
    Parameters:
    -----------
    params_flat : np.ndarray
        Flattened parameter vector
    data : DataFrame
        Training data
    A, D : object arrays
        Fixed observation and prior matrices
    policies : list
        Policy list from agent
    num_actions : list
        Number of actions per factor
    arch_config : dict
        Architecture configuration
    Z_fixed : np.ndarray or None
        Fixed assignment matrix (if not learning Z)
    learn_Z : bool
        Whether to learn Z
    posteriors : np.ndarray or None
        Pre-computed posterior distributions (for speed)
        
    Returns:
    --------
    neg_log_lik : float
        Negative log-likelihood (for minimization)
    """
    # Unpack parameters
    if learn_Z:
        phi0_logits = params_flat[0:2]
        xi0_logits = params_flat[2:4]
        gamma0 = np.exp(params_flat[4])
        
        phi1_logits = params_flat[5:7]
        xi1_logits = params_flat[7:9]
        gamma1 = np.exp(params_flat[9])
        
        Z_logits = params_flat[10:14].reshape(2, 2)
        Z = np.array([softmax(Z_logits[0]), softmax(Z_logits[1])])
    else:
        phi0_logits = params_flat[0:2]
        xi0_logits = params_flat[2:4]
        gamma0 = np.exp(params_flat[4])
        
        phi1_logits = params_flat[5:7]
        xi1_logits = params_flat[7:9]
        gamma1 = np.exp(params_flat[9])
        
        Z = Z_fixed
    
    # Build profiles
    profiles = [
        {'phi_logits': phi0_logits, 'xi_logits': xi0_logits, 'gamma': gamma0},
        {'phi_logits': phi1_logits, 'xi_logits': xi1_logits, 'gamma': gamma1}
    ]
    
    # Create value function
    value_fn = make_value_fn_M3(profiles, Z, policies, num_actions, arch_config)
    
    # Compute log-likelihood
    total_log_lik = 0.0
    
    for idx in range(len(data)):
        obs_surprisal_idx = int(data.iloc[idx]['surprisal_idx'])
        obs_length_idx = int(data.iloc[idx]['length_idx'])
        actual_switch = int(data.iloc[idx]['cs_binary'])
        
        # Use pre-computed posteriors if available, otherwise compute
        if posteriors is not None:
            posterior = posteriors[idx]
        else:
            # State inference
            likelihood_surprisal = A[0][obs_surprisal_idx, :]
            likelihood_length = A[1][obs_length_idx, :]
            prior = D[0]
            posterior = likelihood_surprisal * likelihood_length * prior
            posterior = posterior / (posterior.sum() + 1e-12)
        
        # Get value function parameters
        C_t, E_t, gamma_t = value_fn(posterior, idx)
        
        # Compute action probabilities
        if E_t is not None:
            policy_logits = np.log(E_t + 1e-12) + gamma_t * np.log(C_t[2] + 1e-12)
        else:
            policy_logits = gamma_t * np.log(C_t[2] + 1e-12)
        
        action_probs = softmax(policy_logits)
        prob_actual = action_probs[actual_switch]
        
        total_log_lik += np.log(prob_actual + 1e-12)
    
    return -total_log_lik


def fit_M3_parameters(data, A, D, Z_init, policies, num_actions, arch_config,
                     learn_Z=False, n_restarts=3, verbose=True, warm_start_params=None):
    """
    Fit M3 parameters using gradient-based optimization
    
    Parameters:
    -----------
    data : DataFrame
        Training data
    A, D : object arrays
        Fixed observation and prior matrices
    Z_init : np.ndarray
        Initial assignment matrix
    policies : list
        Policy list
    num_actions : list
        Number of actions per factor
    arch_config : dict
        Architecture configuration
    learn_Z : bool
        Whether to learn Z or keep it fixed
    n_restarts : int
        Number of random restarts
    verbose : bool
        Print progress
    warm_start_params : dict or None
        Parameters from previous fold for warm starting
        
    Returns:
    --------
    best_params : dict
        Best parameters found
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"FITTING M3 PARAMETERS (learn_Z={learn_Z})")
        print(f"{'='*60}")
    
    # Pre-compute posteriors once (major speedup!)
    start_time = time.time()
    posteriors = precompute_posteriors(data, A, D)
    precompute_time = time.time() - start_time
    if verbose:
        print(f"Pre-computed posteriors in {precompute_time:.2f}s")
    
    best_nll = np.inf
    best_params_flat = None
    
    for restart in range(n_restarts):
        if verbose:
            print(f"\nRestart {restart+1}/{n_restarts}")
        
        # Initialize parameters
        if restart == 0 and warm_start_params is not None:
            # Use warm start from previous fold
            if verbose:
                print("  Using warm start from previous fold...")
            profiles = warm_start_params['profiles']
            phi0_init = profiles[0]['phi_logits']
            xi0_init = profiles[0]['xi_logits']
            gamma0_init = np.log(profiles[0]['gamma'])
            phi1_init = profiles[1]['phi_logits']
            xi1_init = profiles[1]['xi_logits']
            gamma1_init = np.log(profiles[1]['gamma'])
        elif restart == 0:
            # Sensible initialization
            phi0_init = np.array([0.5, -0.5])
            phi1_init = np.array([-0.5, 0.5])
            xi0_init = np.array([0.5, -0.5])
            xi1_init = np.array([-0.5, 0.5])
            gamma0_init = np.log(1.5)
            gamma1_init = np.log(0.8)
        else:
            # Random initialization
            phi0_init = np.random.randn(2) * 0.5
            phi1_init = np.random.randn(2) * 0.5
            xi0_init = np.random.randn(2) * 0.5
            xi1_init = np.random.randn(2) * 0.5
            gamma0_init = np.log(np.random.uniform(0.5, 2.0))
            gamma1_init = np.log(np.random.uniform(0.5, 2.0))
        
        if learn_Z:
            Z_logits_init = np.log(Z_init + 1e-12)
            params_init = np.concatenate([
                phi0_init, xi0_init, [gamma0_init],
                phi1_init, xi1_init, [gamma1_init],
                Z_logits_init.flatten()
            ])
        else:
            params_init = np.concatenate([
                phi0_init, xi0_init, [gamma0_init],
                phi1_init, xi1_init, [gamma1_init]
            ])
        
        # Optimize
        if verbose:
            print("  Optimizing...")
        
        opt_start = time.time()
        result = minimize(
            compute_log_likelihood_M3,
            params_init,
            args=(data, A, D, policies, num_actions, arch_config, 
                  Z_init if not learn_Z else None, learn_Z, posteriors),
            method='L-BFGS-B',
            options={'maxiter': 500, 'disp': False, 'ftol': 1e-6}
        )
        opt_time = time.time() - opt_start
        
        if verbose:
            print(f"  Negative LL: {result.fun:.2f}")
            print(f"  Success: {result.success}")
            print(f"  Time: {opt_time:.2f}s")
            print(f"  Iterations: {result.nit}")
        
        if result.fun < best_nll:
            best_nll = result.fun
            best_params_flat = result.x
            if verbose:
                print(f"  *** New best! ***")
    
    # Unpack best parameters
    if learn_Z:
        phi0 = best_params_flat[0:2]
        xi0 = best_params_flat[2:4]
        gamma0 = np.exp(best_params_flat[4])
        phi1 = best_params_flat[5:7]
        xi1 = best_params_flat[7:9]
        gamma1 = np.exp(best_params_flat[9])
        Z_logits = best_params_flat[10:14].reshape(2, 2)
        Z_learned = np.array([softmax(Z_logits[0]), softmax(Z_logits[1])])
    else:
        phi0 = best_params_flat[0:2]
        xi0 = best_params_flat[2:4]
        gamma0 = np.exp(best_params_flat[4])
        phi1 = best_params_flat[5:7]
        xi1 = best_params_flat[7:9]
        gamma1 = np.exp(best_params_flat[9])
        Z_learned = Z_init
    
    best_params = {
        'profiles': [
            {'phi_logits': phi0, 'xi_logits': xi0, 'gamma': gamma0},
            {'phi_logits': phi1, 'xi_logits': xi1, 'gamma': gamma1}
        ],
        'Z': Z_learned,
        'nll': best_nll,
        'total_log_lik': -best_nll
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print("LEARNED PARAMETERS:")
        print(f"{'='*60}")
        print("\nProfile 0 (Low-load):")
        print(f"  phi_logits: {phi0}")
        print(f"  phi_probs:  {softmax(phi0)}")
        print(f"  xi_logits:  {xi0}")
        print(f"  gamma:      {gamma0:.3f}")
        print("\nProfile 1 (High-load):")
        print(f"  phi_logits: {phi1}")
        print(f"  phi_probs:  {softmax(phi1)}")
        print(f"  xi_logits:  {xi1}")
        print(f"  gamma:      {gamma1:.3f}")
        print("\nAssignment Matrix Z:")
        print(f"  low_load:  {Z_learned[0]}")
        print(f"  high_load: {Z_learned[1]}")
    
    return best_params
