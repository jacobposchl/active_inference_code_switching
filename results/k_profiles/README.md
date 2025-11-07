# K-Profiles Experiment

## Overview

This experiment tests M3 with different numbers of profiles (k=1, 2, 3, 4, 5) to determine the optimal number of cognitive profiles for modeling code-switching behavior.

## What It Does

For each k value:
1. Creates k cognitive profiles with different precision and preference parameters
2. Trains M3 using 5-fold cross-validation
3. Evaluates accuracy, log-likelihood, AIC, and BIC
4. Times the training process

## How to Run

### Option 1: Using the batch file (Windows)
```bash
run_k_profiles.bat
```

### Option 2: Direct Python execution
```bash
# Activate virtual environment
.venv\Scripts\activate

# Run experiment
python scripts\profile_number_sweep.py
```

## Expected Runtime

- **k=1:** ~5-10 minutes
- **k=2:** ~10-20 minutes (current default)
- **k=3:** ~15-30 minutes
- **k=4:** ~20-40 minutes
- **k=5:** ~25-50 minutes

**Total: 1-2 hours** (varies by machine)

## Configuration

Edit `scripts/profile_number_sweep.py` to modify:

```python
k_values = [1, 2, 3, 4, 5]      # Which k values to test
n_folds = 5                      # Number of CV folds
n_restarts = 3                   # Optimization restarts per fold
learn_Z = False                  # Whether to learn Z matrix
```

## Outputs

Results are saved to `results/k_profiles/`:

### 1. `k_profiles_comparison.png`
Six-panel visualization showing:
- Accuracy vs k
- Log-likelihood vs k
- AIC and BIC vs k
- **BIC Elbow Plot** (key result for optimal k)
- Model complexity (# parameters) vs k
- Training time vs k

### 2. `k_profiles_summary.csv`
Summary table with all metrics:
```
k, accuracy_mean, accuracy_std, loglik_mean, loglik_std, AIC, BIC, n_params, training_time_min
1, 0.5000, 0.0000, -409.23, 0.55, 4094.34, 4100.33, 1, 8.5
2, 0.5593, 0.0118, -403.15, 1.62, 4050.43, 4110.33, 20, 21.8
...
```

### 3. `k_profiles_results.pkl`
Complete results object containing:
- Full fold-by-fold results for each k
- Learned parameters for each k
- Configuration details

## Interpreting Results

### Key Metric: BIC (Bayesian Information Criterion)

**Lower BIC = Better model**

BIC balances fit quality with model complexity:
- Raw accuracy can increase with k due to overfitting
- BIC penalizes extra parameters
- The **elbow point** in the BIC curve indicates optimal k

### Expected Outcome

Based on theoretical predictions:
- **k=1:** Baseline (no profile differentiation)
- **k=2:** Should have best BIC (validates dual-mode theory)
- **k>2:** May improve fit but worse BIC (overfitting)

### What This Tells Us

If k=2 has the best BIC:
✅ **Validates binary cognitive processing theory**
✅ Fluent vs Effortful modes are sufficient
✅ Additional profiles don't capture meaningful variance

If k>2 is better:
⚠️ May indicate more complex cognitive processes
⚠️ Should analyze learned profiles for interpretability

## Example Results Format

```
============================================================
FINAL COMPARISON
============================================================

  k |     Accuracy |      LogLik |        AIC |        BIC |  Params |  Time(s)
--------------------------------------------------------------------------------
  1 | 0.5000 ± 0.0000 | -409.23 ± 0.55 |    4094.34 |    4100.33 |       1 |   510.23
  2 | 0.5593 ± 0.0118 | -403.15 ± 1.62 |    4050.43 |    4110.33 |      20 |  1310.45
  3 | 0.5621 ± 0.0125 | -401.87 ± 1.71 |    4025.74 |    4150.88 |      29 |  1875.67
  4 | 0.5635 ± 0.0131 | -401.23 ± 1.82 |    4018.46 |    4185.43 |      38 |  2401.23
  5 | 0.5642 ± 0.0138 | -400.95 ± 1.89 |    4015.90 |    4224.56 |      47 |  2950.34

============================================================
BEST k BY CRITERION
============================================================
Best Accuracy:      k=5 (0.5642)
Best Log-Lik:       k=5 (-400.95)
Best AIC:           k=5 (4015.90)
Best BIC (optimal): k=2 (4110.33)  ← KEY RESULT
```

## Troubleshooting

### Out of Memory
If you run out of memory:
```python
n_folds = 3           # Reduce from 5
n_restarts = 2        # Reduce from 3
k_values = [1, 2, 3]  # Test fewer k values
```

### Taking Too Long
For faster testing:
```python
k_values = [1, 2, 3]  # Skip k=4,5
n_folds = 3           # Reduce folds
```

### Import Errors
Ensure you're in the project root:
```bash
cd d:\cs_with_aif
python scripts\profile_number_sweep.py
```

## Next Steps

After running this experiment:

1. **Analyze the BIC elbow plot** - Which k is optimal?
2. **Check profile interpretability** - Run `scripts/analyze_profiles.py` (to be implemented)
3. **Compare to baselines** - Is optimal k=2 M3 better than M1/M2?
4. **Publication narrative** - BIC validation strengthens theoretical claims

## Related Experiments

- **Profile Characterization** (`analyze_profiles.py`) - Understand what profiles learned
- **Ablation Study** (`run_ablation_study.py`) - Test which M3 components matter
- **Feature Attribution** (`compare_m3_lr2.py`) - Compare M3 to logistic regression

## References

See `FUTURE_EXPERIMENTS.md` for complete experimental roadmap.

---

**Author:** Jacob Poschl (jposchl@ucsc.edu)  
**Last Updated:** November 6, 2025
