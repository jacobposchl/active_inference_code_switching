"""
Value Functions for M1, M2, and M3 Models
Each model implements a different precision control strategy
"""

import numpy as np
from pymdp import utils
from .active_inference import softmax


def make_value_fn_M1(config, arch_config):
    """
    Model 1: Static Global Precision
    - Fixed outcome preferences (C)
    - Fixed policy precision (gamma)
    - No adaptation to cognitive state
    
    Parameters:
    -----------
    config : dict
        M1 configuration
    arch_config : dict
        Architecture configuration
        
    Returns:
    --------
    value_fn : function
        Value function for M1
    """
    gamma_fixed = config['gamma_fixed']
    num_modalities = arch_config['num_modalities']
    num_obs_surprisal = len(arch_config['obs_surprisal_labels'])
    num_obs_length = len(arch_config['obs_length_labels'])
    
    # Neutral switch preference
    C_switch_logits = np.array([0.0, 0.0])
    C_switch = softmax(C_switch_logits)
    
    def value_fn(q_state_t, t):
        """Return fixed C and gamma regardless of beliefs"""
        C_t = utils.obj_array(num_modalities)
        C_t[0] = np.ones(num_obs_surprisal) / num_obs_surprisal
        C_t[1] = np.ones(num_obs_length) / num_obs_length
        C_t[2] = C_switch
        
        E_t = None
        gamma_t = gamma_fixed
        
        return C_t, E_t, gamma_t
    
    return value_fn


def make_value_fn_M2(config, arch_config):
    """
    Model 2: Dynamic Global Precision (Entropy-Coupled)
    - Fixed outcome preferences (C)
    - Gamma adapts based on belief entropy
    - High entropy (uncertainty) → lower gamma (more exploration)
    - Low entropy (confidence) → higher gamma (more exploitation)
    
    Parameters:
    -----------
    config : dict
        M2 configuration
    arch_config : dict
        Architecture configuration
        
    Returns:
    --------
    value_fn : function
        Value function for M2
    """
    gamma_base = config['gamma_base']
    k = config['k']
    num_modalities = arch_config['num_modalities']
    num_obs_surprisal = len(arch_config['obs_surprisal_labels'])
    num_obs_length = len(arch_config['obs_length_labels'])
    
    C_switch_logits = np.array([0.0, 0.0])
    C_switch = softmax(C_switch_logits)
    
    def gamma_entropy_coupled(q_state_t):
        """Lower precision when uncertain (higher entropy)"""
        p = np.clip(np.asarray(q_state_t, float), 1e-12, 1.0)
        H = -(p * np.log(p)).sum()  # Entropy
        return gamma_base / (1.0 + k * H)
    
    def value_fn(q_state_t, t):
        C_t = utils.obj_array(num_modalities)
        C_t[0] = np.ones(num_obs_surprisal) / num_obs_surprisal
        C_t[1] = np.ones(num_obs_length) / num_obs_length
        C_t[2] = C_switch
        
        E_t = None
        gamma_t = gamma_entropy_coupled(q_state_t)
        
        return C_t, E_t, gamma_t
    
    return value_fn


def map_action_prefs_to_policy_prefs(xi_logits_per_action, policies, num_actions_per_factor):
    """
    Map action-level preferences to policy-level preferences
    
    Each policy is a sequence of actions. We score each policy by summing
    the preference logits of its constituent actions.
    
    Parameters:
    -----------
    xi_logits_per_action : array
        Preferences for each primitive action
    policies : list of arrays
        Each policy is [policy_len, num_factors] array of action indices
    num_actions_per_factor : list
        Number of actions per control factor
        
    Returns:
    --------
    policy_prefs : array
        Preference logit for each policy
    """
    num_policies = len(policies)
    policy_prefs = np.zeros(num_policies)
    
    for pol_idx, policy in enumerate(policies):
        score = 0.0
        for t in range(len(policy)):
            action_offset = 0
            for f in range(len(num_actions_per_factor)):
                action_idx = policy[t, f]
                global_action_idx = action_offset + action_idx
                score += xi_logits_per_action[global_action_idx]
                action_offset += num_actions_per_factor[f]
        policy_prefs[pol_idx] = score
    
    return policy_prefs


def make_value_fn_M3(profiles, Z, policies, num_actions_per_factor, arch_config):
    """
    Model 3: Profile-Based Precision Control
    - Multiple profiles, each with own phi (outcome prefs), xi (policy prefs), gamma
    - Assignment matrix Z maps states to profiles
    - Effective parameters are belief-weighted mixture
    
    This is the core innovation: different cognitive states (low_load vs high_load)
    activate different profiles (fluent vs effortful), which have different precisions.
    
    Parameters:
    -----------
    profiles : list of dicts
        Each profile has 'phi_logits', 'xi_logits', 'gamma'
    Z : array
        Assignment matrix mapping states to profiles [num_states, num_profiles]
    policies : list of arrays
        Policy list from pymdp agent
    num_actions_per_factor : list
        Number of actions per control factor
    arch_config : dict
        Architecture configuration
        
    Returns:
    --------
    value_fn : function
        Value function for M3
    """
    K = len(profiles)  # Number of profiles
    num_modalities = arch_config['num_modalities']
    num_obs_surprisal = len(arch_config['obs_surprisal_labels'])
    num_obs_length = len(arch_config['obs_length_labels'])
    
    # Extract profile parameters
    PHI = np.array([softmax(p['phi_logits']) for p in profiles])
    GAM = np.array([p['gamma'] for p in profiles])
    
    # Convert action-level xi to policy-level preferences
    if 'xi_logits' in profiles[0] and profiles[0]['xi_logits'] is not None:
        XI_raw = np.array([p['xi_logits'] for p in profiles])
        XI = np.array([map_action_prefs_to_policy_prefs(xi, policies, num_actions_per_factor)
                       for xi in XI_raw])
    else:
        XI = None
    
    def value_fn(q_state_t, t):
        """
        Compute belief-weighted mixture of profiles
        
        Steps:
        1. Compute profile weights: w = q_state @ Z
        2. Mix outcome preferences: phi_t = sum_k w[k] * PHI[k]
        3. Mix precision: gamma_t = sum_k w[k] * GAM[k]
        4. Mix policy preferences: xi_t = sum_k w[k] * XI[k] (if using xi)
        5. Convert to probabilities
        """
        # Profile weights based on current state beliefs
        w = np.asarray(q_state_t, float) @ Z
        w = w / (w.sum() + 1e-12)
        
        # Mix outcome preferences for switch modality
        phi_t = (w[:, None] * PHI).sum(axis=0)
        C_switch = softmax(phi_t)
        
        # Mix precision
        gamma_t = float((w * GAM).sum())
        
        # Build full C vector
        C_t = utils.obj_array(num_modalities)
        C_t[0] = np.ones(num_obs_surprisal) / num_obs_surprisal
        C_t[1] = np.ones(num_obs_length) / num_obs_length
        C_t[2] = C_switch
        
        # Mix policy preferences if available
        if XI is not None:
            xi_t = (w[:, None] * XI).sum(axis=0)
            E_t = softmax(xi_t)
        else:
            E_t = None
        
        return C_t, E_t, gamma_t
    
    return value_fn


def test_value_functions(value_fn_M1, value_fn_M2, value_fn_M3=None):
    """
    Test value functions with different belief states
    
    Parameters:
    -----------
    value_fn_M1, value_fn_M2, value_fn_M3 : functions
        Value functions to test
    """
    print("\n" + "="*60)
    print("TESTING VALUE FUNCTIONS")
    print("="*60)
    
    test_beliefs = [
        np.array([0.9, 0.1]),  # Strongly believe in low_load
        np.array([0.5, 0.5]),  # Uncertain
        np.array([0.1, 0.9])   # Strongly believe in high_load
    ]
    
    # Test M1
    print("\nM1 (Static Global):")
    q_test = test_beliefs[0]
    C_test, E_test, gamma_test = value_fn_M1(q_test, 0)
    print(f"  Gamma: {gamma_test:.3f} (fixed)")
    print(f"  C[2] (switch prefs): {C_test[2]}")
    
    # Test M2
    print("\nM2 (Entropy-Coupled):")
    for q_test in test_beliefs:
        C_test, E_test, gamma_test = value_fn_M2(q_test, 0)
        entropy = -(q_test * np.log(q_test + 1e-12)).sum()
        print(f"  Beliefs: {q_test}, Entropy: {entropy:.3f}, Gamma: {gamma_test:.3f}")
    
    # Test M3 if provided
    if value_fn_M3 is not None:
        print("\nM3 (Profile-Based):")
        for q_test in test_beliefs:
            C_test, E_test, gamma_test = value_fn_M3(q_test, 0)
            print(f"  Beliefs: {q_test}")
            print(f"    Gamma: {gamma_test:.3f}")
            print(f"    C[2] (switch prefs): {C_test[2]}")
    
    print("\n" + "="*60)
    print("VALUE FUNCTION TESTS COMPLETE")
    print("="*60)
