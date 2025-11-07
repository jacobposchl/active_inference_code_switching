"""
Profile Management for M3 Model
Handles profile initialization, learning, and manipulation
"""

import numpy as np
from .active_inference import softmax


class ProfileManager:
    """Manages profiles for M3 model"""
    
    def __init__(self, config):
        """
        Initialize profile manager
        
        Parameters:
        -----------
        config : dict
            M3 configuration with initial profiles and Z matrix
        """
        self.config = config
        self.num_profiles = config['num_profiles']
        self.profiles = [p.copy() for p in config['initial_profiles']]
        self.Z = config['initial_Z'].copy()
    
    def get_profiles(self):
        """Return current profiles"""
        return self.profiles
    
    def get_Z(self):
        """Return current assignment matrix"""
        return self.Z
    
    def set_profiles(self, profiles):
        """Set profiles from learned parameters"""
        self.profiles = [p.copy() for p in profiles]
    
    def set_Z(self, Z):
        """Set assignment matrix from learned parameters"""
        self.Z = Z.copy()
    
    def print_profiles(self):
        """Print current profile parameters"""
        print("\n" + "="*60)
        print("PROFILE PARAMETERS")
        print("="*60)
        
        for i, profile in enumerate(self.profiles):
            name = "Fluent (low_load)" if i == 0 else "Effortful (high_load)"
            print(f"\nProfile {i} ({name}):")
            print(f"  phi_logits (outcome prefs): {profile['phi_logits']}")
            print(f"  phi (probabilities): {softmax(profile['phi_logits'])}")
            if 'xi_logits' in profile:
                print(f"  xi_logits (action prefs): {profile['xi_logits']}")
            print(f"  gamma (precision): {profile['gamma']:.3f}")
        
        print("\n" + "="*60)
        print("Assignment Matrix Z:")
        print("="*60)
        print("                Profile_0  Profile_1")
        print("                (Fluent)   (Effortful)")
        print(f"low_load          {self.Z[0,0]:.2f}       {self.Z[0,1]:.2f}")
        print(f"high_load         {self.Z[1,0]:.2f}       {self.Z[1,1]:.2f}")
    
    def flatten_parameters(self, learn_Z=False):
        """
        Flatten profiles and Z into parameter vector for optimization
        
        Parameters:
        -----------
        learn_Z : bool
            Whether to include Z in parameter vector
            
        Returns:
        --------
        params_flat : np.ndarray
            Flattened parameter vector
        """
        params = []
        
        for profile in self.profiles:
            params.extend(profile['phi_logits'])
            if 'xi_logits' in profile:
                params.extend(profile['xi_logits'])
            params.append(np.log(profile['gamma']))  # Log space for positivity
        
        if learn_Z:
            # Z in log-space (will be converted via softmax)
            params.extend(np.log(self.Z + 1e-12).flatten())
        
        return np.array(params)
    
    def unflatten_parameters(self, params_flat, learn_Z=False):
        """
        Reconstruct profiles and Z from flat parameter vector
        
        Parameters:
        -----------
        params_flat : np.ndarray
            Flattened parameter vector
        learn_Z : bool
            Whether Z is included in parameter vector
            
        Returns:
        --------
        profiles : list
            List of profile dictionaries
        Z : np.ndarray
            Assignment matrix
        """
        profiles = []
        idx = 0
        
        for i in range(self.num_profiles):
            profile = {}
            profile['phi_logits'] = params_flat[idx:idx+2]
            idx += 2
            profile['xi_logits'] = params_flat[idx:idx+2]
            idx += 2
            profile['gamma'] = np.exp(params_flat[idx])  # Convert from log space
            idx += 1
            profiles.append(profile)
        
        if learn_Z:
            Z_logits = params_flat[idx:].reshape(2, 2)
            Z = np.array([softmax(Z_logits[0]), softmax(Z_logits[1])])
        else:
            Z = self.Z.copy()
        
        return profiles, Z
    
    def initialize_for_optimization(self, restart_idx=0):
        """
        Initialize parameters for optimization with optional randomization
        
        Parameters:
        -----------
        restart_idx : int
            Restart index (0 = sensible init, >0 = random init)
            
        Returns:
        --------
        params_init : np.ndarray
            Initial parameter vector
        """
        if restart_idx == 0:
            # Sensible initialization based on data
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
        
        params_init = np.concatenate([
            phi0_init, xi0_init, [gamma0_init],
            phi1_init, xi1_init, [gamma1_init]
        ])
        
        return params_init


def create_profile_manager(config):
    """
    Factory function to create ProfileManager
    
    Parameters:
    -----------
    config : dict
        M3 configuration
        
    Returns:
    --------
    manager : ProfileManager
        Initialized profile manager
    """
    manager = ProfileManager(config)
    manager.print_profiles()
    return manager
