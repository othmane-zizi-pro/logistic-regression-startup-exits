# COMPLETE NOTEBOOK WALKTHROUGH: Startup Exits Analysis

## OVERALL GOAL
This notebook predicts whether a startup will be acquired (successful exit) based on:
- Location (state)
- Market/industry
- Funding characteristics (total funding, timing, types)
- Company age

You build 3 progressively sophisticated logistic regression models and compare them.

---

## SUMMARY OF THE 3 MODELS

### Model 1: Baseline
- **Features**: state, market, funding total, founded year, first/last funding years
- **Purpose**: Simple baseline to establish performance floor
- **Expected**: Moderate performance, may miss important patterns

### Model 2: Feature Engineering
- **Features**: Adds `years_before_first_funding` and `funding_duration`
- **Purpose**: Capture funding dynamics, not just raw dates
- **Expected**: Better fit than Model 1 (lower AIC/BIC)

### Model 3: SMOTE + Funding Types
- **Features**: Adds `primary_funding_type`, uses SMOTE resampling
- **Purpose**: Address class imbalance, incorporate funding strategy
- **Expected**: Best at identifying acquisitions (higher recall)

---

## COMPLETE STEP-BY-STEP BREAKDOWN

### SECTION 1: SETUP (Cells 1-5)

**Cell 1-3: Import Libraries**
- pandas/numpy for data manipulation
- sklearn for modeling and evaluation
- statsmodels for logistic regression (GLM)
- patsy for formula-based modeling

**Cell 5: Load Data**
```python
investments = pd.read_csv("investments_VC.csv")
```
- Contains startup funding and acquisition data
- Target: `acquired` (1 = acquired, 0 = not acquired)

---

### SECTION 2: DATA CLEANING (Cells 6-29)

**Cells 6-13: EDA**
- Inspect data shape, types, missing values
- Understand distribution of target variable
- Identify data quality issues

**Cells 14-18: Column/Row Selection**
- Keep only relevant features
- Remove rows with critical missing data
- Focus on complete cases

**Cells 20-23: Clean Categorical Variables**
- Standardize `state_code` (e.g., "California" → "CA")
- Clean `market` categories
- Remove rare/inconsistent categories

**Cells 24-29: Feature Engineering (Initial)**
- Convert `funding_total_usd` from string to numeric
- Parse dates → extract years
- Create `first_funding_year`, `last_funding_year`
- Result: `investments_clean` dataset

---

### SECTION 3: MODEL 1 - BASELINE (Cells 30-34)

**Cell 32: Define Formula**
```python
acquired ~ C(state_code) + C(market) +
           funding_total_usd + founded_year +
           first_funding_year + last_funding_year
```
- `C()` = treat as categorical (creates dummy variables)

**Cell 32: Build Model**
1. `dmatrices()` creates X (features) and y (target)
2. Remove zero-variance columns
3. Clean column names for readability
4. Train/test split (70/30, stratified)
5. Fit logistic regression: `result = model.fit()`
6. Print model summary (coefficients, p-values)
7. Calculate odds ratios
8. Evaluate on test set
9. Plot ROC curve

**Cell 33: Store Metrics**
```python
aic_1 = result.aic
bic_1 = result.bic
```

**Key Outputs:**
- Model summary table (coefficients, p-values)
- Odds ratio table (interpretable effects)
- Confusion matrix
- Accuracy & ROC-AUC
- ROC curve plot

---

### SECTION 4: MODEL 2 - ENHANCED (Cells 35-43)

**Cells 37-40: Additional Feature Engineering**
- Calculate `years_before_first_funding` = first_funding_year - founded_year
- Calculate `funding_duration` = last_funding_year - first_funding_year
- These capture funding dynamics

**Cell 41: Updated Formula**
```python
acquired ~ C(state_code) + C(market) +
           funding_total_usd + founded_year +
           years_before_first_funding + funding_duration
```
- Replaces raw years with engineered timing features

**Cell 41: Fit Model 2**
- Same process as Model 1
- Train on enhanced features

**Cell 42: Store Metrics**
```python
aic_2 = result.aic
bic_2 = result.bic
```

**Expected Result:** Lower AIC/BIC than Model 1 (better fit)

---

### SECTION 5: MODEL 3 - SMOTE (Cells 44-62)

**Problem Being Solved:**
- Class imbalance: Far more non-acquired than acquired startups
- Model biased toward predicting "not acquired"
- Poor at detecting acquisitions (what we care about!)

**Cells 48-52: Re-introduce Funding Types**
- Add columns: `angel`, `venture`, `seed`, `round_a`, etc.
- Calculate `primary_funding_type` = which type contributed most $

**Cells 54-57: VIF Check (Multicollinearity)**

**Cell 55: Build Design Matrix**
```python
y_prepared, X_prepared = dmatrices(formula, data=model2_cleaned)
```

**Cell 56: Calculate VIF** (NEW - I added this!)
```python
vif_data = pd.DataFrame()
vif_data['Feature'] = X_prepared.columns
vif_data['VIF'] = [variance_inflation_factor(X_prepared.values, i)
                   for i in range(X_prepared.shape[1])]
```

**What is VIF?**
- Variance Inflation Factor measures multicollinearity
- **VIF < 5**: Good (low correlation)
- **VIF 5-10**: Moderate correlation
- **VIF > 10**: Problem! Features are redundant

**Why check before SMOTE?**
- SMOTE creates synthetic data
- Inflates correlations artificially
- VIF on SMOTE data would be misleading

**Cell 57: Visualize VIF**
- Bar chart showing VIF for each feature
- Helps identify problematic correlations

**Cell 58: Linearity Check**
- Verify logistic regression assumptions
- Plot continuous features vs. log-odds

**Cells 60-61: Apply SMOTE**

**What is SMOTE-NC?**
- **S**ynthetic **M**inority **O**ver-sampling **T**echnique
- **NC** = works with both **N**ominal (categorical) and **C**ontinuous data

**How SMOTE works:**
1. Find k-nearest neighbors for minority class (acquired=1)
2. Create synthetic samples between each sample and neighbors
3. For continuous features: Interpolate values
4. For categorical features: Use most common value

**Why SMOTE is better than duplication:**
- Duplicating just copies existing samples → overfitting
- SMOTE creates new samples → better generalization

**CRITICAL: Apply SMOTE to training set ONLY**
```python
X_train, X_test, y_train, y_test = train_test_split(...)
# Apply SMOTE to X_train, y_train only
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
# Test set stays original distribution!
```

**Cell 61: Fit Model 3**
```python
acquired ~ C(state_code) + C(market) +
           funding_total_usd + founded_year +
           years_before_first_funding + funding_duration +
           C(primary_funding_type)
```
- Adds primary funding type
- Trains on SMOTE-balanced data

**Cell 62: Store Metrics**
```python
aic_3 = result.aic
bic_3 = result.bic
```

**Expected Result:**
- May have lower overall accuracy (predicts more acquisitions)
- Better **recall** for acquisitions (fewer false negatives)
- Higher **ROC-AUC** (better discrimination)

---

### SECTION 6: COMPARISON (Cell 65+)

**Cell 65: Create Comparison Table**
```python
comparison = pd.DataFrame({
    'Model': ['Model 1', 'Model 2', 'Model 3'],
    'AIC': [aic_1, aic_2, aic_3],
    'BIC': [bic_1, bic_2, bic_3]
})
comparison['Best_AIC'] = comparison['AIC'] == comparison['AIC'].min()
comparison['Best_BIC'] = comparison['BIC'] == comparison['BIC'].min()
```

**How to interpret:**
- **Lower AIC/BIC = Better model**
- AIC balances fit vs. complexity
- BIC penalizes complexity more (prefers simpler models)
- `Best_AIC` / `Best_BIC` columns show winner

**Typical Results:**
- Model 1: Baseline (highest AIC/BIC)
- Model 2: Better fit (lower AIC/BIC than Model 1)
- Model 3: Best overall or best for minority class detection

---

## KEY CONCEPTS EXPLAINED

### 1. Logistic Regression
- Predicts **probability** of binary outcome
- Formula: `log(p/(1-p)) = β₀ + β₁X₁ + β₂X₂ + ...`
- Coefficients (β) show change in log-odds per unit change in X

**Example:**
- If β for "State: CA" = 0.34
- Being in CA increases log-odds by 0.34
- Odds ratio = exp(0.34) = 1.40
- **Interpretation**: CA startups have 40% higher odds of acquisition

### 2. Train/Test Split
- **Training set (70%)**: Fit model parameters
- **Test set (30%)**: Evaluate generalization
- **Stratified**: Maintains same % acquired in both sets
- **Why?** Prevents overfitting, estimates real-world performance

### 3. Categorical Encoding
- `C(state_code)` creates dummy variables
- One state is reference (baseline = 0)
- Each other state gets a column (1 if that state, 0 otherwise)
- Coefficient shows effect **compared to reference**

**Example:**
```
State: CA     → 1 if California, 0 otherwise
State: NY     → 1 if New York, 0 otherwise
(Reference state like TX has no column)
```

### 4. AIC vs BIC
**Both measure model quality, lower = better**

**AIC** (Akaike Information Criterion):
- Formula: `AIC = -2*log-likelihood + 2*k`
- k = number of parameters
- Balances fit vs. complexity

**BIC** (Bayesian Information Criterion):
- Formula: `BIC = -2*log-likelihood + k*log(n)`
- n = sample size
- Penalizes complexity more than AIC
- Prefers simpler models

**When to use:**
- Use both for model comparison
- If they disagree, consider: Do you prefer simpler (BIC) or better fit (AIC)?

### 5. Class Imbalance Problem

**The Problem:**
```
Not Acquired: 15,000 startups (94%)
Acquired:      1,000 startups (6%)
```

**Naive model:** Always predict "not acquired"
- Accuracy = 94%! (Looks great!)
- But completely useless (never detects acquisitions)

**Why it matters:**
- We care about identifying acquisitions
- High accuracy is misleading
- Need to focus on **recall** for minority class

**SMOTE Solution:**
- Creates synthetic "acquired" examples
- Balances training set to 50/50
- Model learns both classes equally well
- Test on original distribution (realistic evaluation)

### 6. VIF (Variance Inflation Factor)

**What it measures:**
- How much variance of coefficient is inflated due to correlation with other features

**Formula:**
- `VIF_i = 1 / (1 - R²_i)`
- R²_i = R² from regressing feature i on all other features

**Interpretation:**
- **VIF = 1**: No correlation (perfect)
- **VIF < 5**: Acceptable
- **VIF 5-10**: Moderate multicollinearity
- **VIF > 10**: Serious problem

**Example Problem:**
```
first_funding_year: VIF = 25
last_funding_year: VIF = 23
```
These are highly correlated (both measure time)
→ Solution: Use engineered feature `funding_duration` instead

**Why VIF matters:**
- High correlation → unstable coefficients
- Hard to interpret individual effects
- Models may not generalize well

### 7. Odds Ratios

**What they are:**
- `Odds Ratio = exp(coefficient)`
- Multiplicative effect on odds of outcome

**Example:**
```
Funding Total coefficient = 0.0000015
Odds Ratio = exp(0.0000015) ≈ 1.0000015
```
Per $1 increase → 0.00015% higher odds (tiny!)

```
State: CA coefficient = 0.40
Odds Ratio = exp(0.40) = 1.49
```
Being in CA → 49% higher odds

**Interpretation Guide:**
- OR = 1.0: No effect
- OR > 1.0: Increases odds (positive effect)
- OR < 1.0: Decreases odds (negative effect)
- OR = 2.0: Doubles the odds

---

## INTERPRETATION GUIDE: Model Outputs

### Model Summary Table
```
                     coef    std err    z      P>|z|   [0.025   0.975]
Baseline (Intercept) -2.45   0.123    -19.92   0.000   -2.69   -2.21
State: CA             0.34   0.067      5.07   0.000    0.21    0.47
Funding Total (USD)   0.00   0.000      1.19   0.234   -0.00    0.00
```

**How to read:**
1. **coef**: Log-odds coefficient
   - Positive = increases odds of acquisition
   - Negative = decreases odds

2. **P>|z|**: P-value
   - < 0.05 = statistically significant
   - State: CA (0.000) is significant
   - Funding Total (0.234) is NOT significant

3. **[0.025 0.975]**: 95% confidence interval
   - If includes 0 → not significant
   - State: CA [0.21, 0.47] doesn't include 0 → significant

### Confusion Matrix
```
Truth         0      1      ← Actual
Predicted
0          4823     1
1           520     1
↑ Predicted
```

**Reading this:**
- **4823 (True Negative)**: Correctly predicted "not acquired"
- **1 (True Positive)**: Correctly predicted "acquired"
- **520 (False Positive)**: Predicted "acquired" but wasn't
- **1 (False Negative)**: Predicted "not acquired" but was

**Metrics:**
- **Accuracy** = (4823+1) / 5345 = 90.3%
- **Precision** = 1 / (1+520) = 0.2%
- **Recall** = 1 / (1+1) = 50%

**Problem:** Model is terrible at predicting acquisitions!
- Only 1 correct positive prediction
- This is why we need Model 3 with SMOTE

### ROC-AUC Interpretation
- **0.5**: Random guessing (coin flip)
- **0.6-0.7**: Poor model
- **0.7-0.8**: Fair model
- **0.8-0.9**: Good model
- **0.9-1.0**: Excellent model

**Example:**
- Model 1: AUC = 0.65 (poor)
- Model 2: AUC = 0.72 (fair)
- Model 3: AUC = 0.81 (good)
→ Model 3 is best!

---

## VISUAL WORKFLOW

```
DATA LOADING
    ↓
CLEAN & PREPARE
    ↓
┌───────────────────────────────────────┐
│ MODEL 1: BASELINE                     │
│ • state, market, funding, years       │
│ • Train/test split                    │
│ • Fit logistic regression             │
│ • AIC₁, BIC₁                         │
└───────────────┬───────────────────────┘
                ↓
┌───────────────────────────────────────┐
│ MODEL 2: FEATURE ENGINEERING          │
│ • Add timing features:                │
│   - years_before_first_funding        │
│   - funding_duration                  │
│ • AIC₂, BIC₂                         │
└───────────────┬───────────────────────┘
                ↓
┌───────────────────────────────────────┐
│ MODEL 3: SMOTE + FUNDING TYPES        │
│ 1. Calculate primary_funding_type     │
│ 2. VIF check (multicollinearity)      │
│ 3. Apply SMOTE to training set        │
│ 4. Fit on balanced data               │
│ 5. Test on original distribution      │
│ • AIC₃, BIC₃                         │
└───────────────┬───────────────────────┘
                ↓
┌───────────────────────────────────────┐
│ COMPARISON TABLE                      │
│ Model | AIC    | BIC    | Best        │
│ ──────┼────────┼────────┼────────     │
│ 1     | aic_1  | bic_1  |             │
│ 2     | aic_2  | bic_2  |             │
│ 3     | aic_3  | bic_3  | ✓           │
└───────────────────────────────────────┘
```

---

## TROUBLESHOOTING

### Error: "result not defined"
**Cause:** Model cell didn't complete
**Fix:**
1. Run all previous cells first
2. Check if `investments_clean` exists
3. Run model cell individually to see specific error

### Error: "vif_data not defined"
**Cause:** Missing VIF calculation cell
**Fix:** Already fixed! Cell 56 now calculates VIF

### Error: "KeyError: 'primary_funding_type'"
**Cause:** Funding type calculation didn't run
**Fix:** Run cells 48-52 to calculate primary funding type

### Poor Model Performance
**Low accuracy?**
- Check class imbalance → Use Model 3 with SMOTE

**High VIF values (> 10)?**
- Features are too correlated
- Remove or combine correlated features

**Low ROC-AUC?**
- Need more informative features
- Try interaction terms
- Check data quality

### "Run All" Stops Partway
**Causes:**
1. Cell depends on variables from earlier cells
2. Long-running cell appears stuck (be patient!)
3. Error in one cell stops subsequent cells

**Fix:**
- Run cells sequentially to find where it fails
- Check kernel isn't frozen (look for [*] indicator)

---

## QUICK REFERENCE

### Key Variables

| Variable | Type | Description |
|----------|------|-------------|
| `investments` | DataFrame | Raw data from CSV |
| `investments_clean` | DataFrame | Cleaned data (Model 1) |
| `model2_cleaned` | DataFrame | With engineered features (Model 2) |
| `X_train`, `X_test` | DataFrame | Feature matrices |
| `y_train`, `y_test` | Series | Target variable |
| `result` | GLM object | Fitted model |
| `aic_1`, `bic_1` | float | Model 1 metrics |
| `aic_2`, `bic_2` | float | Model 2 metrics |
| `aic_3`, `bic_3` | float | Model 3 metrics |
| `vif_data` | DataFrame | Multicollinearity check |
| `comparison` | DataFrame | Final model comparison |

### Key Functions

```python
# Build design matrices
y, X = dmatrices(formula, data=df)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Fit logistic regression
model = sm.GLM(y_train, X_train, family=sm.families.Binomial())
result = model.fit()

# Calculate VIF
vif = variance_inflation_factor(X.values, i)

# Apply SMOTE
smote = SMOTENC(categorical_features=[...])
X_balanced, y_balanced = smote.fit_resample(X_train, y_train)

# Model metrics
aic = result.aic
bic = result.bic

# Predictions
probs = result.predict(X_test)
preds = (probs >= 0.5).astype(int)
```

---

## WHY EACH STEP MATTERS

1. **Stratified split** → Ensures train/test have same % acquired
2. **Remove zero-variance** → Columns with no variation can't predict
3. **Clean column names** → Makes output readable
4. **Odds ratios** → Easier to interpret than log-odds
5. **Test on original data** → Even if trained on SMOTE, test on real distribution
6. **VIF before SMOTE** → SMOTE inflates correlations
7. **Separate AIC/BIC cells** → Don't re-run expensive model.fit() just for metrics
8. **Store metrics as variables** → Allows comparison without re-running models

---

## EXPECTED RESULTS

After running all cells, you should see:

1. **Three model summaries** with coefficients and p-values
2. **Three AIC/BIC outputs** (one per model)
3. **Confusion matrices** showing predictions vs. actuals
4. **ROC curves** visualizing model discrimination
5. **VIF table and plot** showing multicollinearity
6. **Final comparison table** with Best_AIC and Best_BIC indicators

**Winner:** Model with lowest AIC/BIC (usually Model 3)

---

## FURTHER IMPROVEMENTS

If you wanted to enhance this analysis:

1. **Hyperparameter tuning**: Try different SMOTE sampling strategies
2. **Feature selection**: Use Lasso or stepwise selection
3. **Interaction terms**: Test state×market or funding×timing interactions
4. **Cross-validation**: Use k-fold CV instead of single train/test split
5. **Alternative models**: Try Random Forest, XGBoost for comparison
6. **Threshold optimization**: Find optimal probability cutoff (not just 0.5)
7. **Cost-sensitive learning**: Penalize false negatives more than false positives

---

Good luck with your analysis! 🚀
