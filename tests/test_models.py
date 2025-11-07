"""
Unit Tests for Active Inference Models
Run with: pytest tests/test_models.py
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import ARCHITECTURE_CONFIG, A_MATRIX_CONFIG, M1_CONFIG, M2_CONFIG
from src.models.active_inference import build_A_matrix, build_B_matrix, build_D_matrix, softmax


class TestSoftmax:
    """Test softmax function"""
    
    def test_softmax_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert np.isclose(result.sum(), 1.0)
    
    def test_softmax_all_positive(self):
        x = np.array([-5.0, 0.0, 5.0])
        result = softmax(x)
        assert np.all(result > 0)
    
    def test_softmax_monotonic(self):
        x = np.array([1.0, 2.0, 3.0])
        result = softmax(x)
        assert result[0] < result[1] < result[2]


class TestAMatrix:
    """Test observation model construction"""
    
    def test_A_matrix_shape(self):
        A = build_A_matrix(ARCHITECTURE_CONFIG, A_MATRIX_CONFIG)
        
        # Check number of modalities
        assert len(A) == 3
        
        # Check shapes dynamically based on config
        n_surprisal_bins = len(ARCHITECTURE_CONFIG['obs_surprisal_labels'])
        n_length_bins = len(ARCHITECTURE_CONFIG['obs_length_labels'])
        n_switch_obs = len(ARCHITECTURE_CONFIG['obs_switch_labels'])
        n_states = ARCHITECTURE_CONFIG['num_states']
        
        assert A[0].shape == (n_surprisal_bins, n_states)  # Surprisal
        assert A[1].shape == (n_length_bins, n_states)     # Length
        assert A[2].shape == (n_switch_obs, n_states)      # Switch
    
    def test_A_matrix_normalized(self):
        A = build_A_matrix(ARCHITECTURE_CONFIG, A_MATRIX_CONFIG)
        
        n_states = ARCHITECTURE_CONFIG['num_states']
        
        # Each column should sum to 1
        for modality in [0, 1]:
            for state in range(n_states):
                assert np.isclose(A[modality][:, state].sum(), 1.0)
    
    def test_A_matrix_all_positive(self):
        A = build_A_matrix(ARCHITECTURE_CONFIG, A_MATRIX_CONFIG)
        
        for modality in range(3):
            assert np.all(A[modality] >= 0)


class TestBMatrix:
    """Test transition model construction"""
    
    def test_B_matrix_shape(self):
        B = build_B_matrix(ARCHITECTURE_CONFIG)
        
        n_states = ARCHITECTURE_CONFIG['num_states']
        n_actions = len(ARCHITECTURE_CONFIG['action_labels'])
        
        # One state factor
        assert len(B) == 1
        
        # Shape: [num_states, num_states, num_actions]
        assert B[0].shape == (n_states, n_states, n_actions)
    
    def test_B_matrix_normalized(self):
        B = build_B_matrix(ARCHITECTURE_CONFIG)
        
        n_states = ARCHITECTURE_CONFIG['num_states']
        n_actions = len(ARCHITECTURE_CONFIG['action_labels'])
        
        # Each column (starting state) should sum to 1
        for action in range(n_actions):
            for from_state in range(n_states):
                assert np.isclose(B[0][:, from_state, action].sum(), 1.0)
    
    def test_B_matrix_persistence(self):
        B = build_B_matrix(ARCHITECTURE_CONFIG)
        volatility = ARCHITECTURE_CONFIG['volatility']
        n_actions = len(ARCHITECTURE_CONFIG['action_labels'])
        
        # Diagonal should be (1 - volatility)
        for action in range(n_actions):
            assert np.isclose(B[0][0, 0, action], 1 - volatility)
            assert np.isclose(B[0][1, 1, action], 1 - volatility)


class TestDMatrix:
    """Test prior construction"""
    
    def test_D_matrix_shape(self):
        D = build_D_matrix(ARCHITECTURE_CONFIG)
        
        n_states = ARCHITECTURE_CONFIG['num_states']
        
        assert len(D) == 1
        assert D[0].shape == (n_states,)
    
    def test_D_matrix_normalized(self):
        D = build_D_matrix(ARCHITECTURE_CONFIG)
        
        assert np.isclose(D[0].sum(), 1.0)
    
    def test_D_matrix_uniform(self):
        D = build_D_matrix(ARCHITECTURE_CONFIG)
        
        # Should be uniform prior
        assert np.allclose(D[0], 0.5)


class TestValueFunctions:
    """Test value function creation"""
    
    def test_M1_returns_correct_types(self):
        from src.models.value_functions import make_value_fn_M1
        
        value_fn = make_value_fn_M1(M1_CONFIG, ARCHITECTURE_CONFIG)
        q_state = np.array([0.7, 0.3])
        
        C, E, gamma = value_fn(q_state, 0)
        
        assert len(C) == 3
        assert gamma == M1_CONFIG['gamma_fixed']
    
    def test_M2_gamma_varies_with_entropy(self):
        from src.models.value_functions import make_value_fn_M2
        
        value_fn = make_value_fn_M2(M2_CONFIG, ARCHITECTURE_CONFIG)
        
        # Low entropy (confident)
        q_low_entropy = np.array([0.95, 0.05])
        C1, E1, gamma1 = value_fn(q_low_entropy, 0)
        
        # High entropy (uncertain)
        q_high_entropy = np.array([0.5, 0.5])
        C2, E2, gamma2 = value_fn(q_high_entropy, 0)
        
        # Higher entropy should give lower gamma
        assert gamma2 < gamma1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
