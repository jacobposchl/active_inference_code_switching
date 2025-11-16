"""
Quick verification script for recent config/model fixes.

Checks performed:
- Loads (or preprocesses) data using `src.data_processing`.
- Computes actual bin counts created by `pd.qcut`.
- Builds a runtime `ARCHITECTURE_CONFIG` and corresponding A-matrix using actual bins.
- Calls `initialize_model` to validate A/B/D creation (this will exercise validation checks).
- Constructs a temporary `Agent` using `policy_extraction_gamma` from the runtime config and prints the number of extracted policies.

Run from repo root with: `python scripts/verify_changes.py`
"""

import sys
from pathlib import Path
import pprint

# add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.model_config import DATA_CONFIG, ARCHITECTURE_CONFIG as ARCH_CFG_MODULE
from src.data_processing import full_preprocessing_pipeline, load_processed_data
from src.models.active_inference import initialize_model
from pymdp.agent import Agent
import numpy as np


def main():
    print("=== VERIFY CONFIG & A-MATRIX CHANGES ===")

    processed_path = Path(DATA_CONFIG['processed_data_path'])

    if processed_path.exists():
        print(f"Loading processed data from {processed_path}...")
        model_data = load_processed_data(str(processed_path))
    else:
        raw_path = Path(DATA_CONFIG['raw_data_path'])
        if not raw_path.exists():
            print(f"ERROR: Raw data not found at {raw_path}. Cannot run preprocessing.")
            return 1
        print("Running full preprocessing pipeline (this may take a moment)...")
        model_data, pairs, stats = full_preprocessing_pipeline(str(raw_path), DATA_CONFIG)

    # Compute actual bins
    actual_surprisal_bins = int(model_data['surprisal_idx'].nunique()) if 'surprisal_idx' in model_data.columns else None
    actual_length_bins = int(model_data['length_idx'].nunique()) if 'length_idx' in model_data.columns else None

    print('\nDATA BIN SUMMARY:')
    print(f"  Requested surprisal bins: {DATA_CONFIG.get('n_bins_surprisal')} | Actual surprisal bins: {actual_surprisal_bins}")
    print(f"  Requested length bins:    {DATA_CONFIG.get('n_bins_length')} | Actual length bins:    {actual_length_bins}")

    # Build runtime architecture and A-matrix similar to train_models
    runtime_arch = dict(ARCH_CFG_MODULE)  # shallow copy
    if actual_surprisal_bins is not None:
        runtime_arch['obs_surprisal_labels'] = [f'surp_bin_{i}' for i in range(actual_surprisal_bins)]
    if actual_length_bins is not None:
        runtime_arch['obs_length_labels'] = [f'len_bin_{i}' for i in range(actual_length_bins)]

    n_surp = len(runtime_arch['obs_surprisal_labels'])
    n_len = len(runtime_arch['obs_length_labels'])

    # Build A config
    def _make_a_config(n_surp, n_len):
        surprisal_matrix = np.zeros((n_surp, runtime_arch['num_states']))
        for i in range(n_surp):
            surprisal_matrix[i, 0] = (n_surp - i) / float(n_surp)
            surprisal_matrix[i, 1] = (i + 1) / float(n_surp)
        col_sums = surprisal_matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1.0
        surprisal_matrix = surprisal_matrix / col_sums[None, :]

        length_matrix = np.zeros((n_len, runtime_arch['num_states']))
        for i in range(n_len):
            length_matrix[i, 0] = (n_len - i) / float(n_len)
            length_matrix[i, 1] = (i + 1) / float(n_len)
        col_sums_len = length_matrix.sum(axis=0)
        col_sums_len[col_sums_len == 0] = 1.0
        length_matrix = length_matrix / col_sums_len[None, :]

        return {'surprisal': surprisal_matrix, 'length': length_matrix, 'switch': 'uniform'}

    runtime_a = _make_a_config(n_surp, n_len)

    print('\nInitializing generative model with runtime architecture...')
    try:
        A, B, D = initialize_model(runtime_arch, runtime_a)
    except Exception as e:
        print('\nERROR while initializing model:')
        raise

    # Basic checks
    print('\nBASIC CHECKS:')
    ok = True
    if A[0].shape[0] != n_surp:
        print(f"  ERROR: A[0] rows ({A[0].shape[0]}) != expected surprisal bins ({n_surp})")
        ok = False
    else:
        print(f"  OK: A[0] rows == surprisal bins == {n_surp}")

    if A[1].shape[0] != n_len:
        print(f"  ERROR: A[1] rows ({A[1].shape[0]}) != expected length bins ({n_len})")
        ok = False
    else:
        print(f"  OK: A[1] rows == length bins == {n_len}")

    n_switch = len(runtime_arch['obs_switch_labels'])
    if A[2].shape[0] != n_switch:
        print(f"  ERROR: A[2] rows ({A[2].shape[0]}) != expected switch obs ({n_switch})")
        ok = False
    else:
        print(f"  OK: A[2] rows == switch obs == {n_switch}")

    # Check that columns sum to 1
    for m in range(len(A)):
        sums = A[m].sum(axis=0)
        if not np.allclose(sums, 1.0):
            print(f"  ERROR: modality {m} columns do not sum to 1: {sums}")
            ok = False
        else:
            print(f"  OK: modality {m} columns sum to 1")

    # Extract policies using Agent and check gamma usage (configurable)
    print('\nPOLICY EXTRACTION CHECK:')
    policy_gamma = runtime_arch.get('policy_extraction_gamma', 16.0)
    print(f"  Using policy_extraction_gamma = {policy_gamma}")

    try:
        from pymdp import utils
        C_dummy = utils.obj_array(runtime_arch['num_modalities'])
        C_dummy[0] = np.ones(n_surp) / float(n_surp)
        C_dummy[1] = np.ones(n_len) / float(n_len)
        C_dummy[2] = np.ones(n_switch) / float(n_switch)

        temp_agent = Agent(
            A=A, B=B, C=C_dummy, D=D,
            policy_len=runtime_arch['policy_len'],
            inference_horizon=runtime_arch['inference_horizon'],
            control_fac_idx=runtime_arch['control_fac_idx'],
            use_utility=runtime_arch['use_utility'],
            use_states_info_gain=runtime_arch['use_states_info_gain'],
            action_selection=runtime_arch['action_selection'],
            gamma=policy_gamma
        )
        policies = temp_agent.policies
        print(f"  Extracted {len(policies)} policies (policy_len={runtime_arch['policy_len']})")
    except Exception as e:
        print('  ERROR while extracting policies:')
        raise

    print('\nSUMMARY:')
    if ok:
        print('  All checks passed ✅')
        return 0
    else:
        print('  Some checks failed ❌')
        return 2


if __name__ == '__main__':
    rc = main()
    sys.exit(rc)
