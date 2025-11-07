# Project Directory Structure

```
cs_with_aif/
│
├── config/                          # Configuration Files
│   └── model_config.py             # All hyperparameters & settings
│
├── data/                           # Data Directory
│   ├── raw/                        # Original CSV files (place your data here)
│   │   └── [your_data.csv]
│   └── processed/                  # Cached preprocessed data (auto-generated)
│       └── model_data.pkl
│
├── src/                            # Source Code
│   ├── __init__.py
│   │
│   ├── data_processing.py          # Data Pipeline (379 lines)
│   │   ├── load_raw_data()
│   │   ├── clean_data()
│   │   ├── create_matched_pairs()
│   │   ├── discretize_variables()
│   │   ├── prepare_model_data()
│   │   ├── compute_pair_statistics()
│   │   └── full_preprocessing_pipeline()
│   │
│   ├── models/                     # Model Components
│   │   ├── __init__.py
│   │   │
│   │   ├── active_inference.py     # Core AIF Model (182 lines)
│   │   │   ├── softmax()
│   │   │   ├── build_A_matrix()    # Observation model
│   │   │   ├── build_B_matrix()    # Transition model
│   │   │   ├── build_D_matrix()    # Prior
│   │   │   └── initialize_model()
│   │   │
│   │   ├── value_functions.py      # M1, M2, M3 Models (234 lines)
│   │   │   ├── make_value_fn_M1()  # Static precision
│   │   │   ├── make_value_fn_M2()  # Entropy-coupled
│   │   │   ├── make_value_fn_M3()  # Profile-based ⭐
│   │   │   └── map_action_prefs_to_policy_prefs()
│   │   │
│   │   └── profiles.py             # Profile Management (178 lines)
│   │       └── ProfileManager
│   │           ├── get_profiles()
│   │           ├── set_profiles()
│   │           ├── print_profiles()
│   │           ├── flatten_parameters()
│   │           └── unflatten_parameters()
│   │
│   ├── training/                   # Training Components
│   │   ├── __init__.py
│   │   │
│   │   ├── optimizer.py            # Parameter Learning (224 lines)
│   │   │   ├── compute_log_likelihood_M3()
│   │   │   └── fit_M3_parameters()
│   │   │
│   │   └── cross_validation.py     # CV Framework (118 lines)
│   │       ├── cross_validate_models()
│   │       └── print_cv_results()
│   │
│   └── evaluation/                 # Evaluation & Visualization
│       ├── __init__.py
│       │
│       ├── metrics.py              # Evaluation Metrics (139 lines)
│       │   ├── evaluate_model()    # Main evaluation loop
│       │   └── compare_models()    # Model comparison
│       │
│       └── visualization.py        # Plotting Functions (284 lines)
│           ├── plot_model_comparison()
│           ├── plot_m3_mechanism()
│           └── plot_pair_analysis()
│
├── scripts/                        # Executable Scripts
│   ├── train_models.py            # Main Training Pipeline ⭐ (202 lines)
│   │   # Complete pipeline from data → training → results
│   │
│   └── simple_example.py          # Quick Start Example (76 lines)
│       # Minimal example for single model evaluation
│
├── notebooks/                      # Jupyter Notebooks (for exploration)
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_comparison.ipynb
│   └── 03_sensitivity_analysis.ipynb
│
├── tests/                         # Unit Tests
│   └── test_models.py            # Model Tests (137 lines)
│       ├── TestSoftmax
│       ├── TestAMatrix
│       ├── TestBMatrix
│       ├── TestDMatrix
│       └── TestValueFunctions
│
├── results/                       # Generated Outputs (created automatically)
│   ├── model_comparison.png      # Visualization outputs
│   ├── m3_mechanism.png
│   └── training_results.pkl      # Saved results
│
├── requirements.txt               # Python Dependencies
├── .gitignore                    # Git ignore patterns
├── README.md                     # Main Documentation ⭐
├── PROJECT_SUMMARY.md            # Reorganization Summary
└── QUICK_REFERENCE.md            # Quick Reference Guide

```

## Key Files to Start With

### 🔴 Essential
1. **README.md** - Start here! Complete documentation
2. **config/model_config.py** - All settings in one place
3. **scripts/train_models.py** - Main entry point

### 🟡 Important
4. **scripts/simple_example.py** - Quick start guide
5. **src/data_processing.py** - Understand data pipeline
6. **src/models/value_functions.py** - Core models (M1, M2, M3)

### 🟢 For Customization
7. **src/training/optimizer.py** - Modify learning algorithm
8. **src/evaluation/visualization.py** - Customize plots
9. **tests/test_models.py** - Add your own tests

## Module Dependencies

```
config/model_config.py
    └─> (imported by all modules)

scripts/train_models.py
    ├─> src/data_processing.py
    │   └─> (loads/processes data)
    ├─> src/models/active_inference.py
    │   └─> (builds A, B, D matrices)
    ├─> src/models/value_functions.py
    │   └─> (creates M1, M2, M3)
    ├─> src/models/profiles.py
    │   └─> (manages M3 profiles)
    ├─> src/training/optimizer.py
    │   └─> (learns M3 parameters)
    ├─> src/training/cross_validation.py
    │   └─> (runs CV)
    └─> src/evaluation/
        ├─> metrics.py (evaluates models)
        └─> visualization.py (plots results)
```

## Code Size Breakdown

| Module | Lines | Purpose |
|--------|-------|---------|
| data_processing.py | 379 | Data loading & preparation |
| active_inference.py | 182 | Core generative model |
| value_functions.py | 234 | M1, M2, M3 implementations |
| profiles.py | 178 | M3 profile management |
| optimizer.py | 224 | Parameter learning |
| cross_validation.py | 118 | CV framework |
| metrics.py | 139 | Evaluation |
| visualization.py | 284 | Plotting |
| train_models.py | 202 | Main pipeline |
| test_models.py | 137 | Unit tests |
| **TOTAL** | **~2,077** | **Core functionality** |

## Workflow Diagram

```
┌─────────────────┐
│  Raw Data CSV   │
└────────┬────────┘
         │
         v
┌─────────────────────────┐
│  data_processing.py     │
│  - Load & clean         │
│  - Create pairs         │
│  - Discretize variables │
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│  Processed Data (PKL)   │
└────────┬────────────────┘
         │
         v
┌─────────────────────────┐
│  active_inference.py    │
│  - Build A, B, D        │
└────────┬────────────────┘
         │
         v
┌──────────────────────────────────────┐
│  value_functions.py                  │
│  ┌──────┐  ┌──────┐  ┌──────────┐  │
│  │  M1  │  │  M2  │  │  M3      │  │
│  │Static│  │Entropy│  │Profiles  │  │
│  └──────┘  └──────┘  └──────────┘  │
└──────────────┬───────────────────────┘
               │
               v
┌──────────────────────────────────────┐
│  training/                           │
│  - optimizer.py (learn M3 params)    │
│  - cross_validation.py (CV)          │
└──────────────┬───────────────────────┘
               │
               v
┌──────────────────────────────────────┐
│  evaluation/                         │
│  - metrics.py (accuracy, log-lik)    │
│  - visualization.py (plots)          │
└──────────────┬───────────────────────┘
               │
               v
┌──────────────────────────────────────┐
│  results/                            │
│  - model_comparison.png              │
│  - m3_mechanism.png                  │
│  - training_results.pkl              │
└──────────────────────────────────────┘
```

## Configuration Hierarchy

```
model_config.py
├── DATA_CONFIG
│   ├── raw_data_path
│   ├── n_bins_*
│   └── random_seed
│
├── ARCHITECTURE_CONFIG
│   ├── state_labels
│   ├── obs_*_labels
│   ├── action_labels
│   └── volatility
│
├── A_MATRIX_CONFIG
│   ├── surprisal
│   ├── length
│   └── switch
│
├── M1_CONFIG
│   └── gamma_fixed
│
├── M2_CONFIG
│   ├── gamma_base
│   └── k
│
├── M3_CONFIG
│   ├── num_profiles
│   ├── initial_profiles
│   └── initial_Z
│
├── TRAINING_CONFIG
│   ├── n_folds
│   ├── n_restarts
│   └── learn_Z
│
├── EVAL_CONFIG
│   └── metrics
│
└── VIZ_CONFIG
    ├── figure_size
    ├── dpi
    └── colors
```

## Tips for Navigation

1. **Start Simple**: Run `scripts/simple_example.py` first
2. **Understand Flow**: Read `scripts/train_models.py` top to bottom
3. **Modify Config**: All experiments start in `config/model_config.py`
4. **Debug**: Check `src/evaluation/metrics.py` with `verbose=True`
5. **Extend**: Add functions to relevant module based on purpose

## File Size Reference

- **Small** (<100 lines): `__init__.py` files, simple examples
- **Medium** (100-200 lines): Core modules, specific functionality
- **Large** (200-400 lines): Comprehensive modules with multiple functions
- **Very Large** (>400 lines): Only original monolithic file (2,327 lines)

The modular structure keeps files manageable and focused!
