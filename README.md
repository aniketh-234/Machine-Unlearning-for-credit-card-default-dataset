# Machine Unlearning for Credit Card Default Dataset

## Overview
This project implements **SISA (Sharded, Isolated, Sliced, and Aggregated) Machine Unlearning** on a credit card default prediction dataset. It demonstrates how to efficiently remove individual records from a trained machine learning model without full retraining, ensuring privacy compliance (GDPR, CCPA) while maintaining model performance.

## Research Pipeline

### PHASE 0: Setup, Data & Helpers
- Data loading and preprocessing
- Feature grouping (Profile, History, Account features)
- Train/test split (80/20 stratified split)
- Model factory for creating 3-specialist ensemble
- Helper functions for ensemble prediction, metric calculation, and SHAP explanations

### PHASE 1: Shard Size Optimization
- Tests shard sizes: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
- Evaluates tradeoff between accuracy and unlearning time
- Computes optimization score: Accuracy / Unlearning Time
- Generates 3 visualization plots: Accuracy vs Shards, Unlearning Time vs Shards, Score vs Shards
- Selects optimal NUM_SHARDS automatically

### PHASE 2: Learning
- Trains 3-specialist ensemble on optimal shards
- Specialists: Profiler (GaussianNB), Historian (RandomForest), Accountant (LogisticRegression)
- Generates comprehensive metrics report
- Creates confusion matrix and ROC curves
- Computes feature importances and SHAP explanations

### PHASE 3: Unlearning
- Selects a user to forget (index 20000)
- Identifies the shard containing the user
- Retrains ONLY the affected shard (1 of N shards)
- Executes unlearning in milliseconds instead of full retraining seconds
- Validates predictions post-unlearning

### PHASE 4: Comparison
- Side-by-side metrics comparison (Accuracy, Precision, Recall, F1, AUC-ROC)
- Overlaid ROC curves (before and after unlearning)
- Confusion matrices comparison
- Time comparison: Full Training vs SISA Unlearning
- SHAP beeswarm plots comparison
- SHAP mean |SHAP| bar charts
- Signed SHAP delta analysis

### PHASE 5: Deep Analysis
Comprehensive verification and analysis including:
- **5a. Membership Inference Verification**: Proves user is actually forgotten
- **5b. Unlearning Stability Test**: Tests 20 users to verify accuracy retention
- **5c. SISA Speedup Analysis**: Compares SISA vs full retrain
- **5d. Dummy Baseline Comparison**: Validates model beats baseline
- **5e. Threshold Optimization**: Finds optimal decision threshold for F1
- **5f. Top Feature Importances**: Ranks features by Random Forest importance
- **5g. Feature Correlation Heatmap**: Visualizes feature relationships
- **5h. Class Distribution**: Shows default vs non-default split
- **5i. 5-fold Cross-Validation**: Stratified CV with mean ± std metrics
- **5j. Privacy-Utility Frontier**: Pareto curve and elbow point detection

## Architecture

### 3-Specialist Ensemble Design
```
Input Data
├── Profile Features (SEX, EDUCATION, MARRIAGE, AGE)
│   └── Profiler (GaussianNB)
├── History Features (PAY_* columns excluding amounts)
│   └── Historian (RandomForestClassifier, 50 estimators)
└── Account Features (All others including amounts)
    └── Accountant (LogisticRegression)

Output: Average of 3 specialist probabilities → Ensemble prediction
```

### SISA Strategy
- **Sharded**: Data split into N independent shards
- **Isolated**: Each shard has its own specialist ensemble
- **Sliced**: Ensemble predictions averaged across shards
- **Aggregated**: Final prediction is mean of all shard outputs

## Key Features

### Feature Groups
- **Profile Features**: SEX, EDUCATION, MARRIAGE, AGE (demographics)
- **History Features**: PAY_2, PAY_3, ..., PAY_9 (payment status history)
- **Account Features**: LIMIT_BAL, AGE, BILL_AMT*, PAY_AMT*, MONTHS_BALANCE (financial)

### Model Pipeline
- **Preprocessing**: OneHotEncoder for categorical, StandardScaler for numerical
- **Models**: GaussianNB (profile), RandomForest 50 trees (history), LogisticRegression (account)
- **Ensemble**: Average probability across 3 specialists, mean across shards

## Installation

```bash
# Required packages
pip install pandas numpy matplotlib seaborn scikit-learn shap

# For Colab (if needed)
!pip install shap -q
```

## Usage

Run the code in Google Colab or Jupyter Notebook:

1. Upload your credit card dataset (CSV format with semicolon separator)
2. Execute the notebook cell-by-cell
3. Each phase outputs metrics, visualizations, and analysis

## Key Results

### Performance Metrics
- **Accuracy**: ~0.78-0.82 depending on shard optimization
- **Precision**: ~0.45-0.50 (defaults are minority class)
- **Recall**: ~0.55-0.65 (crucial for identifying defaults)
- **AUC-ROC**: ~0.75-0.80

### Unlearning Performance
- **SISA Speedup**: 10-100× faster than full retrain (depends on NUM_SHARDS)
- **Efficiency Gain**: 90-98% reduction in unlearning time vs full retraining
- **Accuracy Stability**: >99% of original accuracy retained post-unlearning
- **Membership Inference**: Forgotten user confidence shifts toward 0.5 (non-member)

### Feature Importance (Top Predictors)
1. PAY_* delay status columns (strongest predictors)
2. BILL_AMT* variables (payment amounts)
3. PAY_AMT* variables (paid amounts)
4. Age and credit limit

## Dependencies

```python
# Data processing
pandas, numpy

# Visualization
matplotlib, seaborn, shap

# Machine Learning
scikit-learn (GaussianNB, RandomForestClassifier, LogisticRegression)

# Utilities
google.colab (for file upload in Colab)
```

## Privacy Guarantees

✓ **Right to be Forgotten**: User data completely removed from model via SISA
✓ **Efficiency**: Unlearning in O(1/N) time vs full retraining
✓ **Membership Inference**: Forgotten users indistinguishable from non-members
✓ **GDPR/CCPA Compliant**: Supports regulatory data deletion requirements

## Citation

If you use this code in your research, please cite:

```
Machine Unlearning for Credit Card Default Prediction using SISA
Author: Aniketh-234
Repository: github.com/aniketh-234/Machine-Unlearning-for-credit-card-default-dataset
```

## License

This project is open source and available under the MIT License.

## Author

**Aniketh-234** - Implementation and research pipeline design