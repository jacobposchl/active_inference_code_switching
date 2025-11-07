# Future Experiments Plan

## Overview
This document outlines planned experiments to maximize the theoretical significance and interpretability of the Active Inference code-switching models. The focus is on validating that M3's learned profiles correspond to meaningful cognitive states, rather than just optimizing predictive accuracy.

---

## Primary Experiment: Profile Validation & Characterization

**Priority:** ⭐⭐⭐ HIGHEST

### Research Question
Do the learned precision profiles actually capture distinct cognitive processing modes (fluent vs. effortful), or are they just overfitting noise?

### Hypothesis
- **Profile 1 (Fluent):** Low surprisal, shorter sentences, low cognitive load, higher code-switching rate
- **Profile 2 (Effortful):** High surprisal, longer sentences, high cognitive load, lower code-switching rate

### Methodology

#### 1.1 Extract Profile Assignments
- For each sentence in the dataset, determine which profile is active
- Use learned Z-matrix (state-to-profile mapping) and inferred states

#### 1.2 Characterize Profile-Specific Features
Compare the following metrics between profiles:

**Linguistic Features:**
- Mean surprisal of first code-switch word
- Mean translation surprisal
- Mean sentence length
- Sentence complexity (if available)

**Behavioral Features:**
- Code-switching rate (proportion of sentences with switches)
- Position of code-switch in sentence
- Switch frequency patterns

**Computational Features:**
- Mean belief entropy
- Mean precision (gamma) values
- Prediction confidence

#### 1.3 Statistical Validation
- T-tests or Mann-Whitney U tests for feature differences between profiles
- Effect sizes (Cohen's d)
- Visualization: Violin plots, density plots for each feature by profile

#### 1.4 Profile Stability Analysis
- Cross-validation consistency: Do same sentences get same profile assignments across folds?
- Speaker-level analysis: Do individual speakers show profile preferences?
- Temporal stability: Are profiles consistent within speakers over time?

### Expected Outcomes
✅ Profiles show statistically significant differences in cognitive load indicators  
✅ Profile assignments are stable across CV folds (>80% consistency)  
✅ Profiles align with psycholinguistic theory of processing fluency

### Deliverables
- `scripts/analyze_profiles.py` - Analysis script
- `results/profile_characterization.png` - Visualization of profile differences
- `results/profile_statistics.csv` - Statistical comparison table

---

## Experiment 2: Ablation Study

**Priority:** ⭐⭐⭐ HIGH

### Research Question
Which components of M3 drive its improved performance: precision modulation, outcome preferences, or both?

### Model Variants

#### M3a: Precision Control Only
- Profile-based gamma (precision) modulation
- Fixed/uniform outcome preferences (C)
- Fixed policy priors (E)

#### M3b: Outcome Preferences Only
- Profile-based outcome preferences (C)
- Fixed gamma across all states
- Profile-based policy priors (E)

#### M3c: Full M3 (Current)
- Profile-based gamma
- Profile-based C and E
- All components modulated by profile

### Methodology
1. Implement M3a and M3b variants in `src/models/value_functions.py`
2. Train all three variants using same CV framework
3. Compare accuracy, log-likelihood, AIC, BIC
4. Analyze which component contributes most to improvement over M1/M2

### Expected Outcomes
✅ Precision control (M3a) should show majority of improvement  
✅ Combined model (M3c) should perform best but not drastically better than M3a  
✅ Validates that **precision modulation** is the key mechanism

### Deliverables
- `src/models/value_functions.py` - Updated with M3a, M3b variants
- `scripts/run_ablation_study.py` - Automated ablation analysis
- `results/ablation_comparison.png` - Performance comparison
- `results/ablation_results.csv` - Detailed metrics

---

## Experiment 3: Number of Profiles Analysis

**Priority:** ⭐⭐ MEDIUM-HIGH

### Research Question
Is k=2 profiles optimal, or do additional profiles capture more cognitive variability?

### Methodology
1. Train M3 with k = 1, 2, 3, 4, 5 profiles
2. Use same CV framework for each k
3. Compare:
   - Predictive accuracy
   - Log-likelihood
   - BIC (penalizes complexity)
   - Training time
4. Analyze learned profiles for each k

### Expected Outcomes
✅ k=1: Baseline (no profile differentiation)  
✅ k=2: Best BIC (validates binary cognitive modes)  
✅ k>2: Marginal accuracy gains but overfitting (worse BIC)  
✅ Validates theoretical assumption of dual-mode processing

### Deliverables
- `scripts/profile_number_sweep.py` - Automated k-sweep
- `results/k_profiles_comparison.png` - Elbow plot (BIC vs k)
- `results/optimal_k_analysis.md` - Interpretation document

---

## Experiment 4: Feature Attribution Comparison (M3 vs LR2)

**Priority:** ⭐⭐ MEDIUM

### Research Question
Do M3's learned precision patterns align with LR2's feature weights? If so, M3 provides mechanistic explanation of what LR2 learned.

### Methodology

#### 4.1 Extract LR2 Coefficients
```python
lr2_surprisal_weight = β₁
lr2_length_weight = β₂
```

#### 4.2 Extract M3 Patterns
- Correlation between surprisal states and learned gamma values
- Correlation between length states and learned gamma values
- Direction of effects (positive/negative)

#### 4.3 Compare Mechanisms
- Do both models increase/decrease switching probability with surprisal the same way?
- Do the learned profiles' gamma values correlate with LR2's weights?
- Visualization: Overlay M3 precision curves on LR2 decision boundaries

### Expected Outcomes
✅ M3's precision modulation correlates with LR2's feature importance  
✅ Both models capture same underlying patterns  
✅ M3 provides **interpretable mechanism** for LR2's black-box weights

### Deliverables
- `scripts/compare_m3_lr2.py` - Feature attribution analysis
- `results/m3_vs_lr2_mechanisms.png` - Mechanism comparison visualization
- `results/feature_alignment.csv` - Correlation statistics

---

## Experiment 5: Continuous Features Extension

**Priority:** ⭐ MEDIUM-LOW (Complex Implementation)

### Research Question
Can M3 achieve LR2-level accuracy if it operates on continuous features rather than discretized bins?

### Challenges
- Active Inference framework assumes discrete states
- Would require fundamental architecture changes
- May require Gaussian process or continuous state-space extensions

### Methodology (if pursued)
1. Implement continuous state representation (e.g., Gaussian beliefs)
2. Adapt transition and observation models for continuous variables
3. Compare M3_continuous vs M3_discrete vs LR2

### Expected Outcomes
✅ M3_continuous should approach or match LR2 accuracy  
✅ Validates that discretization is the main bottleneck  
⚠️ May lose interpretability of discrete cognitive states

### Deliverables
- `src/models/active_inference_continuous.py` - New model class
- `results/continuous_vs_discrete.png` - Performance comparison

---

## Experiment 6: Bin Size Sensitivity Analysis

**Priority:** ⭐ LOW (Less Theoretical Value)

### Research Question
How does discretization granularity affect M3 performance?

### Methodology
- Test n_bins = [2, 3, 5, 7, 10] for both surprisal and length
- Full grid search: 5×5 = 25 configurations
- Compare accuracy, training time, interpretability

### Expected Outcomes
✅ More bins → better accuracy (approaching LR2)  
⚠️ More bins → worse interpretability  
⚠️ Diminishing returns after 5-7 bins

### Note
**Low priority** because:
- Predictable results (more bins = better accuracy)
- Doesn't add theoretical insight
- High computational cost
- May lose cognitive interpretability

### Deliverables
- `scripts/bin_size_sweep.py` - Automated grid search
- `results/bin_size_heatmap.png` - 2D heatmap of accuracy vs bins

---

## Experiment 7: Speaker-Level Analysis

**Priority:** ⭐⭐ MEDIUM (If speaker IDs available)

### Research Question
Do individual speakers have consistent profile preferences? Can we predict speaker-level code-switching behavior?

### Methodology
1. Group data by speaker ID (if available in dataset)
2. Analyze profile assignment distributions per speaker
3. Test if some speakers are consistently "fluent" vs "effortful"
4. Train speaker-specific M3 models

### Expected Outcomes
✅ Individual differences in profile usage  
✅ Some speakers may be predominantly one profile  
✅ Validates profiles capture stable cognitive traits

### Deliverables
- `scripts/speaker_analysis.py` - Speaker-level analysis
- `results/speaker_profiles.png` - Profile distribution by speaker
- `results/speaker_consistency.csv` - Within-speaker stability metrics

---

## Implementation Timeline

### Phase 1: Core Interpretability (Weeks 1-2)
1. ✅ Profile Validation & Characterization ⭐⭐⭐
2. ✅ Ablation Study ⭐⭐⭐

### Phase 2: Model Validation (Weeks 3-4)
3. ✅ Number of Profiles Analysis ⭐⭐
4. ✅ Feature Attribution Comparison ⭐⭐

### Phase 3: Extensions (Weeks 5-6, if time permits)
5. Speaker-Level Analysis ⭐⭐ (if applicable)
6. Bin Size Sensitivity ⭐ (if needed for completeness)
7. Continuous Features ⭐ (research direction, not immediate)

---

## Success Criteria

### For Publication
To make a strong contribution, we need to demonstrate:

1. ✅ **Interpretability:** Profiles correspond to meaningful cognitive states
2. ✅ **Mechanism:** M3 provides mechanistic explanation of code-switching
3. ✅ **Validation:** Profile structure is optimal (k=2) and stable
4. ✅ **Theoretical Value:** Precision control is the key mechanism (ablation)
5. ✅ **Comparison:** M3 captures same patterns as LR2 but interpretably

### Minimum Viable Results
- Profile characterization showing significant differences
- Ablation study confirming precision control matters
- k=2 validated as optimal via BIC

### Ideal Complete Results
- All of above PLUS
- Speaker-level validation
- M3-LR2 mechanism alignment
- Clear narrative: "M3 explains *how* surprisal affects code-switching through precision control"

---

## Notes

### Current Model Performance
- **M1/M2:** 50% accuracy (chance level)
- **M3:** 55.93% accuracy (11.9% relative improvement)
- **LR2:** 57.52% accuracy (best overall)

### Key Insight
We're not trying to beat LR2 on accuracy. We're trying to show that **M3 provides a mechanistically interpretable explanation** of the cognitive processes that LR2 captures only through feature weights.

---

## Questions to Address

Before starting experiments, clarify:
1. ✅ Is speaker ID available in the dataset? → Enables Experiment 7
2. ✅ Do we have syntactic complexity measures? → Enriches profile characterization
3. ✅ What's the computational budget? → Determines feasibility of grid searches
4. ✅ Primary publication venue? → Cognitive science vs ML conference determines priorities

---

## Files to Create

```
scripts/
  ├── analyze_profiles.py           # Experiment 1
  ├── run_ablation_study.py         # Experiment 2
  ├── profile_number_sweep.py       # Experiment 3
  ├── compare_m3_lr2.py             # Experiment 4
  ├── speaker_analysis.py           # Experiment 7
  └── bin_size_sweep.py             # Experiment 6

results/
  ├── profile_characterization/
  ├── ablation_study/
  ├── k_profiles/
  └── m3_vs_lr2/
```

---

## Contact

For questions about experimental design or implementation, contact:
jposchl@ucsc.edu

Last Updated: November 6, 2025
