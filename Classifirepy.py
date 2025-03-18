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

#Best Parameter From GridSearch
best_params_ada = {'learning_rate': 0.1, 'n_estimators': 200}
best_gradient_boost_model = {'learning_rate': 0.01, 'max_depth': 3, 'n_estimators': 50}
best_params_XG = {'colsample_bytree': 0.8, 'learning_rate': 0.01, 'max_depth': 4, 'min_child_weight': 1, 'n_estimators': 100, 'subsample': 0.5}
best_params_RF = {'max_depth': 10, 'min_samples_leaf': 4, 'min_samples_split': 10, 'n_estimators': 200}

# Parameters to adjust to reduce overfitting
param_adjustments = {
    'AdaBoost': {'learning_rate': [0.05, 0.1, 0.15], 'n_estimators': [100, 150, 200]},
    'GradientBoosting': {'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5], 'n_estimators': [50, 100, 150]},
    'XGBoost': {'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 4, 5], 'n_estimators': [50, 100, 150], 'subsample': [0.5, 0.7]},
    'RandomForest': {'max_depth': [5, 7, 10], 'min_samples_leaf': [2, 4, 6], 'min_samples_split': [10, 15], 'n_estimators': [100, 150, 200]}
}

# Variables to store all results
results = []


# 10 rounds of testing
for model_name in ['AdaBoost', 'GradientBoosting', 'XGBoost', 'RandomForest']:
    if model_name == 'AdaBoost':
        model = AdaBoostClassifier(**best_params_ada)
        param_grid = param_adjustments['AdaBoost']
    elif model_name == 'GradientBoosting':
        model = GradientBoostingClassifier(**best_gradient_boost_model)
        param_grid = param_adjustments['GradientBoosting']
    elif model_name == 'XGBoost':
        model = XGBClassifier(**best_params_XG)
        param_grid = param_adjustments['XGBoost']
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(**best_params_RF)
        param_grid = param_adjustments['RandomForest']

    # Experiment with customized parameter values ​​in a loop.
    for learning_rate in param_grid.get('learning_rate', [None]):
        for n_estimators in param_grid['n_estimators']:
            for max_depth in param_grid.get('max_depth', [None]):
                for min_samples_leaf in param_grid.get('min_samples_leaf', [1]):
                    for min_samples_split in param_grid.get('min_samples_split', [2]):
                        for subsample in param_grid.get('subsample', [1.0]):
                            # Adjust parameters in each round
                            if model_name == 'AdaBoost':
                                model.set_params(learning_rate=learning_rate, n_estimators=n_estimators)
                            elif model_name == 'GradientBoosting':
                                model.set_params(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth)
                            elif model_name == 'XGBoost':
                                model.set_params(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, subsample=subsample)
                            elif model_name == 'RandomForest':
                                model.set_params(max_depth=max_depth, min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split, n_estimators=n_estimators)

                            # Train Model
                            model.fit(X_train, y_train)

                            # Calculate metrics
                            accuracy_train, accuracy_test, precision, recall, f1,accuracy = calculate_metrics(model, X_train, X_test, y_train, y_test)

                            # Record results
                            result = {
                                'Model': model_name,
                                'Learning Rate': learning_rate,
                                'N Estimators': n_estimators,
                                'Max Depth': max_depth,
                                'Min Samples Leaf': min_samples_leaf,
                                'Min Samples Split': min_samples_split,
                                'Subsample': subsample,
                                'Train Accuracy': accuracy_train,
                                'Test Accuracy': accuracy_test,
                                'accuracy':accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1 Score': f1
                            }
                            results.append(result)

