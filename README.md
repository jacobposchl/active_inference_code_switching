# Active Inference Code-Switching Models

This project implements Active Inference models to predict bilingual code-switching behavior, demonstrating that code-switching is driven by state-specific precision control.

## Project Structure

```
cs_with_aif/
├── config/
│   └── model_config.py          # All hyperparameters and configuration
├── data/
│   ├── raw/                     # Place your CSV files here
│   └── processed/               # Automatically generated processed data
├── src/
│   ├── data_processing.py       # Data loading, cleaning, discretization
│   ├── models/
│   │   ├── active_inference.py  # Core AIF model (A, B, D matrices)
│   │   ├── value_functions.py   # M1, M2, M3 value functions
│   │   └── profiles.py          # Profile management for M3
│   ├── training/
│   │   ├── optimizer.py         # Parameter learning
│   │   └── cross_validation.py  # CV framework
│   └── evaluation/
│       ├── metrics.py           # Evaluation metrics
│       └── visualization.py     # Plotting functions
├── scripts/
│   └── train_models.py          # Main training script
├── notebooks/                    # Jupyter notebooks for exploration
├── results/                      # Generated results and figures
├── tests/                        # Unit tests
└── requirements.txt

```

## Quick Start

### 1. Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Setup

Place your raw data file in `data/raw/`:
```
data/raw/11152019_two_entropies_appended_input_R_v4.csv
```

Or update the path in `config/model_config.py`:
```python
DATA_CONFIG = {
    'raw_data_path': 'your/path/to/data.csv',
    ...
}
```

### 3. Run Training

```bash
cd scripts
python train_models.py
```

This will:
- Preprocess data and create matched pairs
- Initialize three Active Inference models (M1, M2, M3)
- Perform 5-fold cross-validation with parameter learning
- Generate visualizations
- Save results to `results/`

## Data Source

The dataset used in this project is from:

**Calvillo, J., Fang, L., Cole, J., & Reitter, D. (2020).** Surprisal Predicts Code-Switching in Chinese English Bilingual Text. *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*. DOI: [10.18653/v1/2020.emnlp-main.330](https://doi.org/10.18653/v1/2020.emnlp-main.330)

## Models

### M1: Static Global Precision
- Fixed precision parameter (gamma)
- Baseline model with no adaptation

### M2: Surprisal-Coupled Dynamic Precision
- Precision varies with belief uncertainty
- High surprisal → lower precision (exploration)
- Low surprisal → higher precision (exploitation)

### M3: Profile-Based Precision Control (Our Innovation)
- **Two cognitive profiles**: Fluent (low-load) vs Effortful (high-load)
- Each profile has own precision and outcome preferences
- State-specific precision modulation
- **Best performance** - validates core hypothesis

## Configuration

All hyperparameters are in `config/model_config.py`:

```python
# Data processing
DATA_CONFIG = {
    'n_bins_surprisal': 3,  # Number of discretization bins
    'n_bins_length': 3,
    'random_seed': 42
}

# M3 parameters
M3_CONFIG = {
    'num_profiles': 2,
    'initial_profiles': [...],  # Initial profile parameters
    'initial_Z': [...]           # State-to-profile assignment
}

# Training
TRAINING_CONFIG = {
    'n_folds': 5,
    'n_restarts': 3,
    'learn_Z': False
}
```


## Output Files

After training, find results in `results/`:

```
results/
├── model_comparison.png      # All models side-by-side
├── m3_mechanism.png          # M3 internal mechanism
└── training_results.pkl      # Complete results dictionary
```

## Example Usage

### Quick Evaluation (Python)

```python
from config.model_config import *
from src.data_processing import load_processed_data
from src.models.active_inference import initialize_model
from src.models.value_functions import make_value_fn_M1
from src.evaluation.metrics import evaluate_model

# Load data
data = load_processed_data('data/processed/model_data.pkl')

# Initialize model
A, B, D = initialize_model(ARCHITECTURE_CONFIG, A_MATRIX_CONFIG)

# Create value function
value_fn = make_value_fn_M1(M1_CONFIG, ARCHITECTURE_CONFIG)

# Evaluate
results = evaluate_model('M1', value_fn, data, A, B, D, 
                        ARCHITECTURE_CONFIG, verbose=True)

print(f"Accuracy: {results['accuracy']:.4f}")
print(f"Log-Likelihood: {results['total_log_lik']:.2f}")
```

### Custom Model Training

```python
from src.training.optimizer import fit_M3_parameters

# Learn M3 parameters on your data
learned_params = fit_M3_parameters(
    train_data, A, D, Z_init, policies, num_actions,
    ARCHITECTURE_CONFIG, learn_Z=False, n_restarts=5
)

# Access learned parameters
profiles = learned_params['profiles']
Z_matrix = learned_params['Z']
```

## Extending the Framework

### Add a New Model

1. Create value function in `src/models/value_functions.py`:

```python
def make_value_fn_M4(config, arch_config):
    def value_fn(q_state_t, t):
        # Your custom logic here
        C_t = ...  # Outcome preferences
        E_t = ...  # Policy priors
        gamma_t = ...  # Precision
        return C_t, E_t, gamma_t
    return value_fn
```

2. Add config to `config/model_config.py`:

```python
M4_CONFIG = {
    'name': 'M4',
    'description': 'My custom model',
    'custom_param': 1.5,
    'n_params': 3
}
```

3. Evaluate in `scripts/train_models.py`:

```python
value_fn_M4 = make_value_fn_M4(M4_CONFIG, ARCHITECTURE_CONFIG)
results_M4 = evaluate_model('M4', value_fn_M4, model_data, ...)
```

### Modify Observation Model

Edit `A_MATRIX_CONFIG` in `config/model_config.py`:

```python
A_MATRIX_CONFIG = {
    'surprisal': np.array([
        [0.7, 0.1],  # Stronger differentiation
        [0.2, 0.3],
        [0.1, 0.6]
    ]),
    ...
}
```

## Performance Tips

### For Faster Training

1. **Reduce CV folds** in `config/model_config.py`:
```python
TRAINING_CONFIG = {
    'n_folds': 3,  # Instead of 5
    'n_restarts': 2,  # Instead of 3
}
```

2. **Use coarser discretization**:
```python
DATA_CONFIG = {
    'n_bins_surprisal': 2,  # Binary instead of tertile
}
```

3. **Enable parallel processing** (future):
```python
TRAINING_CONFIG = {
    'n_jobs': 4,  # Use 4 CPU cores
}
```

### For Better Accuracy

1. **Finer discretization**:
```python
DATA_CONFIG = {
    'n_bins_surprisal': 5,
    'n_bins_length': 5,
}
```

2. **More optimization restarts**:
```python
TRAINING_CONFIG = {
    'n_restarts': 5,
    'max_iter': 1000,
}
```

3. **Learn assignment matrix**:
```python
TRAINING_CONFIG = {
    'learn_Z': True,
}
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Ensure you're in the project root
cd d:\cs_with_aif

# Install dependencies
pip install -r requirements.txt
```

### Data Not Found

```
ERROR: Raw data file not found
```

Solution: Place CSV in `data/raw/` or update path in `config/model_config.py`

### Memory Issues

If you run out of memory with large datasets:
1. Reduce `n_folds` in training config
2. Process data in batches
3. Use `verbose=False` in evaluation


## Contact

jposchl@ucsc.edu

## Acknowledgments

- Original notebook: Colab implementation
- PyMDP library: Active Inference framework
