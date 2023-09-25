
# XGBoost Model for Technical Indicators (Finding Features with XGBoost)

This repository is dedicated to training and evaluating an XGBoost classifier on the Bitcoin technical indicators dataset. It aims to predict trading signals (like 'long', 'short', or 'neutral') based on the values of various indicators.

## Note
The Signal column, which serves as the target for this model, is generated from the Technical Analysis Repository. Ensure you've processed your data there to obtain the Signal labels before using this model.

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
![Feature Importance Chart](https://github.com/tzelalouzeir/XGBoost_Indicators_2/blob/main/img/normal_xgb_feature.png)
![ROC](https://github.com/tzelalouzeir/XGBoost_Indicators_2/blob/main/img/normal_roc_xgb.png)
![Performance](https://github.com/tzelalouzeir/XGBoost_Indicators_2/blob/main/img/normal_performance.PNG)


## Related Projects

- [Technical Analysis Repository](<https://github.com/tzelalouzeir/XGBoost_Indicators>): This repository fetches 120 days of hourly Bitcoin price data, calculates technical indicators, and analyzes the relations between these indicators.
- [XGBoost Model Optimization](https://github.com/tzelalouzeir/XGBoost_Indicators_3): Optimizing the hyperparameters of an XGBoost classifier using the hyperopt library.
