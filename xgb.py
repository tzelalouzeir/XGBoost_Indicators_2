import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, log_loss

# Signal column, every Dataframe Coming from code https://github.com/tzelalouzeir/XGBoost_Indicators/blob/main/ta_mean.py
# You can save these columns as a csv on currently csv and load this data or merge with ta_mean.py
# This is continue of ta_mean.py

# Remove NaN rows
df.dropna(inplace=True)

# Label encode the target column
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['Signal'])

# Prepare data
X = df.drop(['Signal'], axis=1)
y = y_encoded

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the XGBoost model
clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
clf.fit(X_train, y_train)

# Get feature importances
importances = clf.feature_importances_
feature_names = X.columns

# Sort feature importances in descending order and match the feature names
sorted_indices = np.argsort(importances)[::-1]
sorted_names = [feature_names[i] for i in sorted_indices]

# Plot the feature importances
plt.figure()
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[sorted_indices], align="center")
plt.xticks(range(X.shape[1]), sorted_names, rotation=90)
plt.show()

## performance
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
matrix = confusion_matrix(y_test, y_pred)
probabilities = clf.predict_proba(X_test)[:, 1]  # get the probability of the positive class
auc = roc_auc_score(y_test, probabilities)

print(matrix)
print(report)
print(f"Accuracy: {accuracy:.2f}")
print(f"ROC-AUC Score: {auc:.2f}")
fpr, tpr, thresholds = roc_curve(y_test, probabilities)
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.2f})")
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
