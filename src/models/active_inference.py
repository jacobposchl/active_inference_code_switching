"""
Active Inference Model Components
Builds the core generative model: A (observation), B (transition), D (prior) matrices
"""

import numpy as np
from pymdp import utils


def softmax(x):
    """
    Numerically stable softmax
    
    Parameters:
    -----------
    x : array-like
        Input array
        
    Returns:
    --------
    probs : np.ndarray
        Softmax probabilities
    """
    x = np.asarray(x, dtype=float)
    x_max = np.max(x)
    exp_x = np.exp(x - x_max)
    return exp_x / (exp_x.sum() + 1e-12)


def build_A_matrix(config, a_matrix_config):
    """
    Build observation model A where A[modality][obs, state] = p(obs|state)
    
    Parameters:
    -----------
    config : dict
        Architecture configuration
    a_matrix_config : dict
        A matrix parameters from config
        
    Returns:
    --------
    A : object array
        Observation likelihood matrices for each modality
    """
    num_modalities = config['num_modalities']
    num_states = config['num_states']
    
    A = utils.obj_array(num_modalities)
    
    # Modality 0: Surprisal observations
    A[0] = a_matrix_config['surprisal'].copy()
    # Normalize columns (ensure each column sums to 1)
    A[0] = A[0] / A[0].sum(axis=0, keepdims=True)
    
    # Modality 1: Length observations
    A[1] = a_matrix_config['length'].copy()
    A[1] = A[1] / A[1].sum(axis=0, keepdims=True)
    
    # Modality 2: Switch observations (uniform - determined by action)
    num_obs_switch = len(config['obs_switch_labels'])
    A[2] = np.ones((num_obs_switch, num_states)) / num_obs_switch
    
    return A


def build_B_matrix(config):
    """
    Build transition model B where B[0][s', s, a] = p(s'|s,a)
    
    For code-switching, each sentence is independent, so we model
    persistence with some volatility (cognitive state can change between sentences)
    
    Parameters:
    -----------
    config : dict
        Architecture configuration
        
    Returns:
    --------
    B : object array
        State transition matrices
    """
    num_states = config['num_states']
    num_actions = [len(config['action_labels'])]
    volatility = config['volatility']
    
    B = utils.obj_array(1)  # One state factor
    B[0] = np.zeros((num_states, num_states, num_actions[0]))
    
    # Transitions are same for both actions (actions don't change cognitive state)
    for a in range(num_actions[0]):
        B[0][:, :, a] = np.array([
            [1-volatility, volatility],    # from low_load
            [volatility, 1-volatility]      # from high_load
        ])
    
    return B


def build_D_matrix(config):
    """
    Build initial state prior D where D[0][s] = p(s)
    
    Parameters:
    -----------
    config : dict
        Architecture configuration
        
    Returns:
    --------
    D : object array
        Initial state distribution
    """
    num_states = config['num_states']
    D = utils.obj_array(1)
    D[0] = np.ones(num_states) / num_states  # Uniform prior
    return D


def print_model_architecture(A, B, D, config):
    """
    Print formatted model architecture information
    
    Parameters:
    -----------
    A, B, D : object arrays
        Model matrices
    config : dict
        Architecture configuration
    """
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE")
    print("="*60)
    
    print(f"\nHidden States: {config['state_labels']}")
    print(f"Number of states: {config['num_states']}")
    
    print(f"\nObservation Modalities:")
    print(f"  0. Surprisal: {config['obs_surprisal_labels']}")
    print(f"  1. Length: {config['obs_length_labels']}")
    print(f"  2. Switch: {config['obs_switch_labels']}")
    
    print(f"\nAction Factor:")
    print(f"  0. Language selection: {config['action_labels']}")
    
    print("\n" + "="*60)
    print("A Matrix (Observation Model):")
    print("="*60)
    
    print("\nModality 0 - Surprisal p(obs|state):")
    print("                low_load  high_load")
    for i, label in enumerate(config['obs_surprisal_labels']):
        print(f"  {label:10s}  {A[0][i,0]:.3f}     {A[0][i,1]:.3f}")
    
    print("\nModality 1 - Length p(obs|state):")
    print("                low_load  high_load")
    for i, label in enumerate(config['obs_length_labels']):
        print(f"  {label:10s}  {A[1][i,0]:.3f}     {A[1][i,1]:.3f}")
    
    print("\nB Matrix (Transition Model):")
    print("Action 0 (maintain_chinese):")
    print("           to_low  to_high")
    print(f"from_low    {B[0][0,0,0]:.2f}    {B[0][1,0,0]:.2f}")
    print(f"from_high   {B[0][0,1,0]:.2f}    {B[0][1,1,0]:.2f}")
    
    print("\nAction 1 (switch_to_english):")
    print("           to_low  to_high")
    print(f"from_low    {B[0][0,0,1]:.2f}    {B[0][1,0,1]:.2f}")
    print(f"from_high   {B[0][0,1,1]:.2f}    {B[0][1,1,1]:.2f}")
    
    print(f"\nD Matrix (Initial State Prior): {D[0]}")
    
    print("\n" + "="*60)
    print("MODEL ARCHITECTURE COMPLETE")
    print("="*60)


def initialize_model(arch_config, a_matrix_config):
    """
    Initialize complete generative model
    
    Parameters:
    -----------
    arch_config : dict
        Architecture configuration
    a_matrix_config : dict
        A matrix parameters
        
    Returns:
    --------
    A, B, D : object arrays
        Complete generative model matrices
    """
    print("\nInitializing generative model...")
    
    A = build_A_matrix(arch_config, a_matrix_config)
    B = build_B_matrix(arch_config)
    D = build_D_matrix(arch_config)
    
    print_model_architecture(A, B, D, arch_config)
    
    return A, B, D
