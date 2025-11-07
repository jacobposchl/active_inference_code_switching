# Optimization Guide for M3 Cross-Validation

This guide explains the optimizations made to speed up M3 parameter learning and cross-validation.

## Summary of Optimizations

### 1. **Pre-computed Posteriors** (Major Speedup!)
**Location**: `src/training/optimizer.py` - `precompute_posteriors()`

**What it does**: 
- Computes posterior state distributions for all data points once before optimization
- Avoids redundant Bayesian inference calculations during each optimization iteration

**Expected speedup**: 2-5x faster per optimization run

**How to use**: Automatic - happens when you call `fit_M3_parameters()`

### 2. **Warm Start Between Folds**
**Location**: `src/training/cross_validation.py` and `optimizer.py`

**What it does**:
- Uses learned parameters from fold N to initialize fold N+1
- Reduces optimization time by starting from a good solution

**Expected speedup**: 1.5-2x faster for folds 2-5

**How to use**: 
- Enabled by default in sequential mode (`n_jobs=1`)
- Set `'use_warm_start': True` in `TRAINING_CONFIG`

### 3. **Parallel Cross-Validation**
**Location**: `src/training/cross_validation.py`

**What it does**:
- Runs multiple CV folds in parallel using all CPU cores
- Uses joblib for efficient parallelization

**Expected speedup**: Near-linear with number of cores (4 cores ≈ 3.5x faster)

**Trade-off**: 
- Disables warm-start (each fold starts fresh)
- Best for computers with 4+ cores

**How to use**:
```python
# In config/model_config.py
TRAINING_CONFIG = {
    'n_jobs': -1,  # Use all cores
    # or
    'n_jobs': 4,   # Use 4 cores
}
```

### 4. **Improved Convergence Tolerance**
**Location**: `src/training/optimizer.py`

**What it does**:
- Added `ftol=1e-6` to L-BFGS-B optimizer
- Stops when improvement is negligible

**Expected speedup**: 10-30% fewer iterations

### 5. **Timing Information**
**Location**: Both optimizer and cross-validation modules

**What it does**:
- Reports time for each operation
- Helps identify bottlenecks

## Configuration Options

### In `config/model_config.py`:

```python
TRAINING_CONFIG = {
    'n_folds': 5,           # Number of CV folds
    'n_restarts': 3,        # Random restarts per fold
    'max_iter': 500,        # Max iterations per optimization
    'n_jobs': 1,            # Parallel jobs (1=sequential, -1=all cores)
    'use_warm_start': True, # Warm start (only for n_jobs=1)
}
```

## Performance Tuning Recommendations

### For Fast Iteration (Development)
```python
TRAINING_CONFIG = {
    'n_folds': 3,      # Fewer folds
    'n_restarts': 2,   # Fewer restarts
    'n_jobs': -1,      # Use all cores
}
```
**Expected time**: 2-5 minutes (depending on data size)

### For Best Results (Final Analysis)
```python
TRAINING_CONFIG = {
    'n_folds': 5,      # Standard k-fold
    'n_restarts': 5,   # More restarts for better solution
    'n_jobs': 1,       # Sequential with warm-start
}
```
**Expected time**: 10-20 minutes (but higher quality results)

### For Large Datasets
```python
TRAINING_CONFIG = {
    'n_folds': 5,
    'n_restarts': 3,
    'n_jobs': -1,      # Parallel is better for large data
}
```

## Benchmarking Your System

Run this to test performance:

```python
import time
from config.model_config import *
from src.training.cross_validation import cross_validate_models
from scripts.train_models import *  # Load your data and models

# Test sequential with warm-start
TRAINING_CONFIG['n_jobs'] = 1
start = time.time()
cv_results_seq, _ = cross_validate_models(...)
time_seq = time.time() - start
print(f"Sequential: {time_seq:.1f}s")

# Test parallel
TRAINING_CONFIG['n_jobs'] = -1
start = time.time()
cv_results_par, _ = cross_validate_models(...)
time_par = time.time() - start
print(f"Parallel: {time_par:.1f}s")
print(f"Speedup: {time_seq/time_par:.2f}x")
```

## Expected Performance Gains

| Configuration | Baseline | Optimized | Speedup |
|--------------|----------|-----------|---------|
| 5 folds, 3 restarts (seq) | ~15 min | ~5 min | 3x |
| 5 folds, 3 restarts (parallel, 4 cores) | ~15 min | ~3 min | 5x |
| 5 folds, 5 restarts (seq) | ~25 min | ~8 min | 3x |

*Times are approximate and depend on dataset size and hardware*

## What's Optimized

✅ **Pre-computation of posteriors** - Major win!
✅ **Warm-start between folds** - Good for sequential
✅ **Parallel fold processing** - Good for multi-core
✅ **Early stopping** - Automatic
✅ **Timing instrumentation** - For monitoring

## What's NOT Optimized (Yet)

❌ **The value function itself** - Could be vectorized
❌ **Gradient computation** - Using numerical gradients
❌ **Policy evaluation** - Done for each data point

These could be future optimization targets if needed.

## Monitoring Performance

The optimized code prints timing information:

```
Pre-computed posteriors in 0.15s
Restart 1/3
  Optimizing...
  Negative LL: 234.56
  Success: True
  Time: 2.34s
  Iterations: 45
  *** New best! ***
...
M3 training completed in 7.82s
...
CROSS-VALIDATION COMPLETED in 312.45s (5.21 min)
```

Watch these times to verify optimizations are working!

## Troubleshooting

### "Module 'joblib' not found"
```bash
pip install joblib
```

### Parallel mode is slower than sequential
- You may have a small dataset where overhead dominates
- Try `n_jobs=1` with warm-start instead

### Out of memory errors with parallel mode
- Reduce `n_jobs` to 2 or 4 instead of -1
- Or use sequential mode

### Optimization not converging
- Increase `max_iter` to 1000
- Try more random restarts (`n_restarts=5`)
- Check your data for issues

## Additional Tips

1. **First run is slower**: Python compiles functions on first use
2. **Use processed data cache**: Make sure `processed_data_path` exists
3. **Profile your code**: Use `cProfile` to find other bottlenecks
4. **Consider GPU**: PyMDP doesn't use GPU, but you could port critical sections

## Questions?

Check the code documentation in:
- `src/training/optimizer.py` - Parameter fitting
- `src/training/cross_validation.py` - CV framework
