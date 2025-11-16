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
    
    # Helper for validation/normalization
    def _validate_and_normalize(mat, modality_name, expected_rows=None):
        mat = np.asarray(mat, dtype=float)
        if mat.ndim != 2:
            raise ValueError(f"{modality_name} matrix must be 2D (obs x states); got shape {mat.shape}")
        if mat.shape[1] != num_states:
            raise ValueError(f"{modality_name} matrix has {mat.shape[1]} columns but expected {num_states} (num_states)")
        if expected_rows is not None and mat.shape[0] != expected_rows:
            raise ValueError(f"{modality_name} matrix has {mat.shape[0]} rows but expected {expected_rows} (labels length)")
        # clip tiny negative values due to numerical issues
        mat = np.clip(mat, 0.0, None)
        col_sums = mat.sum(axis=0)
        if np.any(col_sums == 0):
            raise ValueError(f"{modality_name} matrix has a column with zero sum; cannot normalize")
        mat = mat / col_sums[None, :]
        if not np.all(np.isfinite(mat)) or np.any(mat < 0):
            raise ValueError(f"{modality_name} matrix contains invalid values after normalization")
        if not np.allclose(mat.sum(axis=0), 1.0, atol=1e-8):
            raise ValueError(f"{modality_name} matrix columns do not sum to 1 after normalization")
        return mat

    # Modality 0: Surprisal observations
    if 'surprisal' not in a_matrix_config:
        raise KeyError("a_matrix_config must contain key 'surprisal'")
    expected_surprisal_rows = len(config.get('obs_surprisal_labels', [])) or None
    A[0] = _validate_and_normalize(a_matrix_config['surprisal'], 'Surprisal', expected_rows=expected_surprisal_rows)

    # Modality 1: Length observations
    if 'length' not in a_matrix_config:
        raise KeyError("a_matrix_config must contain key 'length'")
    expected_length_rows = len(config.get('obs_length_labels', [])) or None
    A[1] = _validate_and_normalize(a_matrix_config['length'], 'Length', expected_rows=expected_length_rows)

    # Modality 2: Switch observations
    num_obs_switch = len(config.get('obs_switch_labels', ['no_switch', 'switch']))
    if 'switch' in a_matrix_config and not (isinstance(a_matrix_config['switch'], str) and a_matrix_config['switch'] == 'uniform'):
        # Expect an explicit array
        A[2] = _validate_and_normalize(a_matrix_config['switch'], 'Switch', expected_rows=num_obs_switch)
    else:
        # Build explicit uniform switch matrix
        A[2] = np.ones((num_obs_switch, num_states), dtype=float) / float(num_obs_switch)
    
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
    num_states = int(config.get('num_states'))
    action_labels = list(config.get('action_labels', []))
    n_actions = len(action_labels)

    if n_actions == 0:
        raise ValueError("config['action_labels'] must contain at least one action")

    # Volatility: probability of switching away from the current state
    # Accept either a single scalar `volatility` or a list `action_volatilities` with one per action
    volatility = float(config.get('volatility', 0.1))
    if not (0.0 <= volatility <= 1.0):
        raise ValueError(f"volatility must be between 0 and 1; got {volatility}")

    action_vols = config.get('action_volatilities', None)
    if action_vols is None:
        action_vols = [volatility] * n_actions
    else:
        if len(action_vols) != n_actions:
            raise ValueError("length of 'action_volatilities' must equal number of actions")
        action_vols = [float(v) for v in action_vols]
        for v in action_vols:
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"each action_volatility must be between 0 and 1; got {v}")

    # Build B: shape (num_states, num_states, n_actions) where B[s', s, a] = p(s' | s, a)
    B = utils.obj_array(1)
    B[0] = np.zeros((num_states, num_states, n_actions), dtype=float)

    for a_idx, a_vol in enumerate(action_vols):
        if num_states == 1:
            B[0][:, :, a_idx] = np.ones((1, 1), dtype=float)
            continue

        persistence = 1.0 - a_vol
        off_diag = a_vol / float(max(1, num_states - 1))

        mat = np.full((num_states, num_states), off_diag, dtype=float)
        np.fill_diagonal(mat, persistence)

        # Numeric safety: clip tiny negatives, normalize columns to sum to 1
        mat = np.clip(mat, 0.0, 1.0)
        col_sums = mat.sum(axis=0)
        if np.any(col_sums == 0):
            raise ValueError(f"Constructed transition matrix for action {a_idx} has a zero column; check volatility values")
        mat = mat / col_sums[None, :]
        if not np.all(np.isfinite(mat)):
            raise ValueError(f"Non-finite values in transition matrix for action {a_idx}")
        if not np.allclose(mat.sum(axis=0), 1.0, atol=1e-8):
            raise ValueError(f"Transition matrix columns do not sum to 1 for action {a_idx}")

        B[0][:, :, a_idx] = mat

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
