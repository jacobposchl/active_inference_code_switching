"""
Configuration file for Active Inference Code-Switching Models
Contains all hyperparameters and model architecture settings
"""

import numpy as np

# ============================================================
# DATA CONFIGURATION
# ============================================================
DATA_CONFIG = {
    'raw_data_path': 'data/raw/11152019_two_entropies_appended_input_R_v4.csv',
    'processed_data_path': 'data/processed/model_data.pkl',
    'required_columns': [
        'surprisal_first_cs_word_trans',
        'translation_sentence_length',
        'frequency_negative_ln_first_cs_word_trans',
        'entropy_at_cs_point',
        'sent_type'
    ],
    'n_bins_surprisal': 150, #note : higher bins for finer granularity, shows better results
    'n_bins_length': 150, 
    'n_bins_frequency': 150,
    'random_seed': 42
}

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
ARCHITECTURE_CONFIG = {
    # State space
    'state_labels': ['low_load', 'high_load'],
    'num_states': 2,
    
    # Observation space - AUTO-GENERATED from DATA_CONFIG n_bins
    'obs_surprisal_labels': [f'surp_bin_{i}' for i in range(DATA_CONFIG['n_bins_surprisal'])],
    'obs_length_labels': [f'len_bin_{i}' for i in range(DATA_CONFIG['n_bins_length'])],
    'obs_switch_labels': ['no_switch', 'switch'],
    'num_modalities': 3,
    
    # Action space
    'action_labels': ['maintain_chinese', 'switch_to_english'],
    'num_action_factors': 1,
    
    # Transition dynamics
    'volatility': 0.1,  # State transition probability
    
    # Policy configuration
    'policy_len': 1,
    'inference_horizon': 1,
    'control_fac_idx': [0],
    'use_utility': True,
    'use_states_info_gain': False,
    'action_selection': 'stochastic'
}

# ============================================================
# MODEL 1: STATIC GLOBAL PRECISION
# ============================================================
M1_CONFIG = {
    'name': 'M1',
    'description': 'Static global precision',
    'gamma_fixed': 1.2,
    'n_params': 1
}

# ============================================================
# MODEL 2: ENTROPY-COUPLED DYNAMIC PRECISION
# ============================================================
M2_CONFIG = {
    'name': 'M2',
    'description': 'Entropy-coupled dynamic precision',
    'gamma_base': 1.6,
    'k': 1.2,  # Entropy coupling strength
    'n_params': 2
}

# ============================================================
# MODEL 3: PROFILE-BASED PRECISION CONTROL
# ============================================================
M3_CONFIG = {
    'name': 'M3',
    'description': 'Profile-based state-specific precision',
    'num_profiles': 2,
    
    # Initial profile parameters (before learning)
    'initial_profiles': [
        {  # Profile 0: Fluent (low_load)
            'phi_logits': np.array([2.0, -2.0]),  # Strong no-switch preference
            'xi_logits': np.array([1.0, -1.0]),
            'gamma': 2.5
        },
        {  # Profile 1: Effortful (high_load)
            'phi_logits': np.array([-1.0, 1.0]),  # Prefer switch
            'xi_logits': np.array([-0.5, 0.5]),
            'gamma': 0.8
        }
    ],
    
    # Assignment matrix Z: Maps states to profiles
    'initial_Z': np.array([
        [0.9, 0.1],  # When in low_load: mostly use fluent profile
        [0.1, 0.9]   # When in high_load: mostly use effortful profile
    ])
    # Note: n_params is computed dynamically based on number of profiles and state space
}

# ============================================================
# OBSERVATION MODEL (A MATRIX) PARAMETERS
# ============================================================
# Function to generate A matrices based on n_bins
def _generate_observation_model():
    """Generate observation likelihood based on number of bins"""
    n_surp_bins = DATA_CONFIG['n_bins_surprisal']
    n_len_bins = DATA_CONFIG['n_bins_length']
    
    # Surprisal: lower bins more likely in low_load, higher bins in high_load
    surprisal_matrix = np.zeros((n_surp_bins, 2))
    for i in range(n_surp_bins):
        # Linear gradient: low bins favor low_load, high bins favor high_load
        surprisal_matrix[i, 0] = (n_surp_bins - i) / n_surp_bins  # Low load
        surprisal_matrix[i, 1] = (i + 1) / n_surp_bins            # High load
    # Normalize
    surprisal_matrix = surprisal_matrix / surprisal_matrix.sum(axis=0, keepdims=True)
    
    # Length: shorter sentences in low_load, longer in high_load
    length_matrix = np.zeros((n_len_bins, 2))
    for i in range(n_len_bins):
        length_matrix[i, 0] = (n_len_bins - i) / n_len_bins  # Low load
        length_matrix[i, 1] = (i + 1) / n_len_bins            # High load
    # Normalize
    length_matrix = length_matrix / length_matrix.sum(axis=0, keepdims=True)
    
    return {
        'surprisal': surprisal_matrix,
        'length': length_matrix,
        'switch': 'uniform'
    }

A_MATRIX_CONFIG = _generate_observation_model()

# ============================================================
# TRAINING CONFIGURATION
# ============================================================
TRAINING_CONFIG = {
    'n_folds': 5,
    'n_restarts': 3,  # Number of optimization restarts
    'max_iter': 500,
    'optimization_method': 'L-BFGS-B',
    'learn_Z': False,  # Whether to learn assignment matrix
    'split_by_pairs': True,  # Keep matched sentences together in CV
    'n_jobs': -1,  # Number of parallel jobs (1=sequential, -1=all cores)
    'use_warm_start': True,  # Use previous fold params to initialize next fold
}

# ============================================================
# EVALUATION CONFIGURATION
# ============================================================
EVAL_CONFIG = {
    'metrics': ['accuracy', 'log_likelihood', 'aic', 'bic', 'correlation'],
    'save_predictions': True,
    'save_inferred_states': True,
    'verbose': True
}

# ============================================================
# VISUALIZATION CONFIGURATION
# ============================================================
VIZ_CONFIG = {
    'figure_size': (18, 15),
    'dpi': 300,
    'save_format': 'png',
    'results_dir': 'results',
    'colors': {
        'M1': '#1f77b4',
        'M2': '#2ca02c',
        'M3_hand': '#ff7f0e',
        'M3_learned': '#d62728',
        'LR1': '#9467bd',
        'LR2': '#8c564b'
    }
}

# ============================================================
# SENSITIVITY ANALYSIS CONFIGURATION
# ============================================================
SENSITIVITY_CONFIG = {
    'test_n_bins': [2, 3, 4, 5, 6, 7],
    'cv_folds': 3,
    'n_restarts': 2
}
