# Results Summary

## Project Aim

This project reproduces an existing R-based credit risk analysis workflow in Python. The main goal is not only to build machine learning models, but also to make the workflow clear, documented, modular, and reproducible.

## Dataset

The project uses the Loan Approval Classification dataset from Kaggle.

The dataset contains:

- 45,000 observations
- 14 variables
- 0 missing values at the initial loading stage

The target variable is `loan_status`. In this dataset, the positive class represents about 22.22% of all observations.

## Reproduction Workflow

The Python reproduction workflow includes:

- data loading
- exploratory data analysis
- missing value checking
- outlier handling
- data visualization
- categorical variable encoding
- train-test splitting
- Decision Tree classification
- Random Forest classification
- model comparison

The full workflow can be reproduced by running:

`python main.py`

## Main Results

| Model | Test Accuracy |
|---|---:|
| Decision Tree | 90.77% |
| Random Forest | 92.62% |

The Random Forest model achieved the higher test accuracy, with an improvement of about 1.85 percentage points compared with the Decision Tree.

## Confusion Matrices

### Decision Tree

[[8464, 286],
 [752, 1748]]

### Random Forest

[[8521, 228],
 [602, 1898]]

## Note on Random Forest

The Random Forest model reached a training accuracy of 1.00, while the test accuracy was 92.62%. This suggests that the model may overfit the training data, but it still performed better than the Decision Tree on the test set.

Because the Random Forest workflow drops rows with missing values after preprocessing, its test set size is slightly different from the Decision Tree test set. Therefore, the comparison should be understood as a model comparison after each model's preprocessing workflow, not as a comparison on exactly identical test rows.

## Generated Outputs

The project generates exploratory plots, model visualizations, and feature importance plots.

Generated figures are saved in:

`outputs/figures/`

The main output files include:

- `age_by_education.png`
- `numeric_histograms.png`
- `income_home_ownership_gender.png`
- `loan_intent_count.png`
- `decision_tree.png`
- `random_forest_feature_importance.png`

## Final Interpretation

The Random Forest model performed better than the Decision Tree on the test set. The Decision Tree remains useful because it is easier to interpret, while the Random Forest provides stronger predictive performance.

However, the main focus of this project is reproducibility. The project can be cloned from GitHub, dependencies can be installed from `requirements.txt`, and the full workflow can be run through `main.py`.
