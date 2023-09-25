
# XGBoost Model for Bitcoin Technical Indicators

This repository is dedicated to training and evaluating an XGBoost classifier on the Bitcoin technical indicators dataset. It aims to predict trading signals (like 'long', 'short', or 'neutral') based on the values of various indicators.

## Features

1. **Data Cleaning**: Removes rows with NaN values from the dataset.
2. **Data Preprocessing**: Uses label encoding for the target column (Signal) and prepares the training and test data.
3. **XGBoost Classifier**: An XGBoost model is trained on the dataset.
4. **Feature Importance**: Evaluates and visualizes the importance of each feature used in the model.
5. **Model Evaluation**: Evaluates the model's performance using metrics such as accuracy, ROC-AUC score, and more. It also provides a classification report and a confusion matrix.

## Installation

Make sure you have the following Python packages installed:

- `numpy`
- `matplotlib`
- `xgboost`
- `sklearn`

You can install them using `pip`:

```bash
pip install numpy matplotlib xgboost scikit-learn
```

## Usage

1. Clone this repository.
2. Navigate to the repository's root directory in your terminal.
3. Run the code using the following command:

```bash
python xgb.py
```

## Sample Outputs

The code will generate:

- A bar chart showcasing the feature importances.
- A printed confusion matrix, classification report, accuracy score, and ROC-AUC score.
- A ROC curve plot.

## Related Projects

- [Technical Analysis Repository](<[link-to-previous-repo](https://github.com/tzelalouzeir/XGBoost_Indicators)>): This repository fetches 120 days of hourly Bitcoin price data, calculates technical indicators, and analyzes the relations between these indicators.
- [Repo3](<link-to-repo3>): Short description of Repo3.
