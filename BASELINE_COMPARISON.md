# Baseline Comparison Guide

## Overview

This guide shows you how to compare your Active Inference models (M1, M2, M3) against logistic regression baselines.

## Baseline Models

### LR1: Surprisal Only
- Uses only `surprisal_first_cs_word_trans`
- Simplest baseline - just word surprisal
- 2 parameters (coefficient + intercept)

### LR2: Surprisal + Length (Full Model)
- Uses `surprisal_first_cs_word_trans` and `translation_sentence_length`
- Controls for sentence length effects
- Most complete baseline with available features
- 3 parameters

**Note**: LR3 (with frequency) is not included by default because `frequency_negative_ln_first_cs_word_trans` is not in the processed data. To add it, you would need to modify `prepare_model_data()` in `src/data_processing.py` to include this column.

## How to Run

### Option 1: Integrated with Training

The baselines are **automatically included** when you run:
```bash
python scripts/train_models.py
```

The script will:
1. Train M1, M2, M3 models
2. Train LR1, LR2, LR3 baselines on the same CV folds
3. Compare all models
4. Generate comparison plots including baselines

### Option 2: View Comparison After Training

After training, run:
```bash
python scripts/compare_models.py
```

This will:
- Load saved results
- Print detailed comparison tables
- Show which model is best
- Compare best AI model vs best baseline
- Generate CSV table

## Understanding the Output

### During Training (Step 5.5)

```
### STEP 5.5: LOGISTIC REGRESSION BASELINES ###

LOGISTIC REGRESSION BASELINES
============================================================
Models: ['LR1', 'LR2', 'LR3']

Fold 1/5
  LR1 Accuracy: 0.7234
  LR2 Accuracy: 0.7456
  LR3 Accuracy: 0.7512
...

BASELINE MODEL SUMMARY
============================================================

LR1: ['surprisal_first_cs_word_trans']
  Accuracy:  0.7245 ± 0.0123
  Log-Lik:   -234.56 ± 12.34
  AIC:       473.12 ± 24.68
  BIC:       481.23 ± 25.12
  N params:  2

LR2: ['surprisal_first_cs_word_trans', 'translation_sentence_length']
  Accuracy:  0.7456 ± 0.0145
  Log-Lik:   -212.34 ± 10.23
  AIC:       430.68 ± 20.46
  BIC:       442.89 ± 21.34
  N params:  3
...
```

### Comparison Output

```
DETAILED MODEL COMPARISON
================================================================================

Model           Accuracy    Log-Lik        AIC        BIC   N Params
--------------------------------------------------------------------------------

Active Inference Models:
--------------------------------------------------------------------------------
M1              0.7123     -256.78     517.56     525.67          2
M2              0.7234     -245.12     496.24     508.45          3
M3_learned      0.7567     -198.45     416.90     441.23         10

Logistic Regression Baselines:
--------------------------------------------------------------------------------
LR1             0.7245     -234.56     473.12     481.23          2
LR2             0.7456     -212.34     430.68     442.89          3
LR3             0.7512     -205.67     419.34     435.56          4

BEST MODELS
================================================================================
Best Accuracy:       M3_learned      (0.7567)
Best Log-Likelihood: M3_learned      (-198.45)
Best AIC:            M3_learned      (416.90)
Best BIC:            LR3             (419.34)
```

### Active Inference vs Baseline

```
ACTIVE INFERENCE vs BASELINES
================================================================================

Best Active Inference Model: M3_learned
Best Baseline Model:          LR3

Metric              M3_learned             LR3      Difference
--------------------------------------------------------------------------------
Accuracy                0.7567          0.7512         +0.0055
Log-Likelihood       -198.45         -205.67          +7.22
AIC (lower better)    416.90          419.34          -2.44
BIC (lower better)    441.23          435.56          +5.67

INTERPRETATION
================================================================================
✓ Active Inference (M3_learned) has better log-likelihood (+7.22)
✓ Active Inference has better (lower) AIC (-2.44)
✗ Baseline has better (lower) BIC (+5.67)

Active Inference wins 2/3 metrics
```

## What Do These Metrics Mean?

### Accuracy
- Percentage of correct predictions
- Higher is better
- Baseline: 50% (chance)

### Log-Likelihood
- How well the model explains the data
- Higher (less negative) is better
- Raw fit quality

### AIC (Akaike Information Criterion)
- Balances fit quality with model complexity
- **Lower is better**
- Formula: `AIC = 2k - 2*log-likelihood` (k = number of parameters)
- Penalizes complex models

### BIC (Bayesian Information Criterion)
- Similar to AIC but penalizes complexity more strongly
- **Lower is better**
- Formula: `BIC = k*log(n) - 2*log-likelihood` (n = sample size)
- More conservative than AIC

## Interpreting Results

### If Active Inference Wins

✓ **Better Log-Likelihood**: AI model explains data better
✓ **Better AIC/BIC**: Improvement justifies added complexity
→ **Conclusion**: Cognitive model adds value over simple features

### If Baseline Wins

✗ **Better Log-Likelihood**: Simple features explain data just as well
✗ **Better AIC/BIC**: Added complexity not justified
→ **Conclusion**: May need to refine AI model or features

### Mixed Results (Common!)

- M3 might have better log-likelihood but worse BIC
- This means: M3 fits better, but complexity penalty is high
- **Trade-off**: Better explanation vs. parsimony
- Consider: Is the mechanistic insight worth the complexity?

## Customizing Baselines

### Add New Features

Edit `src/models/baselines.py`:
```python
def get_baseline_configs():
    return {
        'LR1': ['surprisal_first_cs_word_trans'],
        'LR2': ['surprisal_first_cs_word_trans', 'translation_sentence_length'],
        'LR3': ['surprisal_first_cs_word_trans', 'translation_sentence_length', 
                'frequency_negative_ln_first_cs_word_trans'],
        'LR4': ['your_new_feature', 'another_feature'],  # Add this
    }
```

### Change Scaling

The default follows Calvillo et al. (2020): mean=0, std=0.5

To change:
```python
results = train_logistic_baseline(
    train_data, test_data, features, 
    scale_std=1.0  # Change to 1.0 for standard scaling
)
```

## Files Generated

After running, you'll find:

```
results/
├── model_comparison.png          # Visual comparison of all models
├── model_comparison.csv          # Table of all results
├── m3_mechanism.png              # M3 mechanism visualization
└── training_results.pkl          # All results (Python object)
```

## Common Questions

**Q: Why does M3 have higher BIC even with better accuracy?**
A: BIC heavily penalizes complex models. M3 has 10 parameters vs LR3's 4. The improved fit may not justify 6 extra parameters by BIC's strict standards.

**Q: Should I use AIC or BIC?**
A: 
- AIC: Better for prediction, less conservative
- BIC: Better for model selection, favors simpler models
- Both are informative! Report both.

**Q: Can I add non-linear features to baselines?**
A: Yes! Add polynomial or interaction terms to baseline_configs. But remember: you're testing if AI's *mechanism* adds value over *any* features.

**Q: What if all baselines beat all AI models?**
A: This suggests the cognitive mechanism isn't captured well. Consider:
- Different value functions
- Different state representations  
- Additional contextual factors

## Next Steps

1. **Run the comparison**: `python scripts/train_models.py`
2. **Check results**: `python scripts/compare_models.py`
3. **Examine plots**: Open `results/model_comparison.png`
4. **Read CSV**: Check `results/model_comparison.csv` for exact numbers
5. **Iterate**: Refine your models based on what you learn!

## References

- Calvillo, J., Brouwer, H., & Crocker, M. W. (2020). Prediction-based learning in native vs. non-native language processing. *Cognitive Science*.
- Akaike, H. (1974). A new look at the statistical model identification. *IEEE Transactions on Automatic Control*.
- Schwarz, G. (1978). Estimating the dimension of a model. *The Annals of Statistics*.
