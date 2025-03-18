# Import Library
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RepeatedStratifiedKFold
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score,f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.tree import export_graphviz
from io import StringIO
from sklearn import preprocessing
from IPython.display import Image
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import statsmodels.api as sm
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
#Import Data
data=pd.read_excel("C:\Users\Asus\OneDrive\เดสก์ท็อป\Main folder\My_Project\Kmeans_Data_Povertygap.xlsx")
X = data[[ 'x2', 'x3','x5', 'x6', 'x7',  'x10','x12','x13']]
y = data[['y']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Imbalance Data
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

X_train=X_resampled
y_train=y_resampled

# Parameter For This Study
best_params_RF = {'max_depth': 5,'min_samples_leaf': 4,'min_samples_split': 10,
                            'n_estimators': 300,'max_features': 'sqrt','bootstrap': False,}
best_params_ada= {'n_estimators':100,'learning_rate':0.05}
best_params_XG= {'colsample_bytree': 1, 'learning_rate': 0.01, 'max_depth': 3, 'min_samples_leaf': 1, 'n_estimators': 50, 'subsample': 0.7,'min_samples_split':2}
best_params_RF= {'max_depth': 5, 'min_samples_leaf': 4, 'min_samples_split': 15, 'n_estimators': 100}

#Buiild Model Random Forest
RF_model = RandomForestClassifier(**best_params_RF,random_state=42)
RF_model.fit(X_train, y_train)

y_pred = RF_model.predict(X_test)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
n_scores = cross_val_score(RF_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print("Cross-validated training scores:", n_scores)
print("Mean CV_train score:", n_scores.mean())
print("Test score:", RF_model.score(X_test, y_test))

accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, average='weighted') * 100
recall = recall_score(y_test, y_pred, average='weighted')*100
f1 = f1_score(y_test, y_pred, average='weighted') * 100

print(f"Accuracy score = {accuracy:.2f}%")
print(f"Precision score = {precision:.2f}%")
print(f"Recall score = {recall:.2f}")
print(f"F1 score = {f1:.2f}%")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')


# GridSearch
XG_model = XGBClassifier(**best_params_XG,random_state=42)
XG_model.fit(X_train, y_train)

# Predict on the test set
y_pred = XG_model.predict(X_test)

# Evaluate the model
cv = RepeatedStratifiedKFold(n_splits=4, n_repeats=3, random_state=42)
n_scores = cross_val_score(XG_model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)
print("Cross-validated training scores:", n_scores)
print("Mean CV_train score: %.4f" % n_scores.mean())
print("Test score: %.4f" % XG_model.score(X_test, y_test))

# Calculate and print evaluation metrics
accuracy = accuracy_score(y_test, y_pred) * 100
precision = precision_score(y_test, y_pred, average='weighted') * 100
recall = recall_score(y_test, y_pred, average='weighted') * 100
f1 = f1_score(y_test, y_pred, average='weighted') * 100

print(f"Accuracy score = {accuracy:.2f}%")
print(f"Precision score = {precision:.2f}%")
print(f"Recall score = {recall:.2f}%")
print(f"F1 score = {f1:.2f}%")

print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm).plot(cmap='Blues')
