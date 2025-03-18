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

#Create Function
def calculate_metrics(model, X_train, X_test, y_train, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    accuracy_train = accuracy_score(y_train, train_pred)
    accuracy_test = accuracy_score(y_test, test_pred)
    precision = precision_score(y_test, test_pred, average='weighted')
    recall = recall_score(y_test, test_pred, average='weighted')
    f1 = f1_score(y_test, test_pred, average='weighted')
    accuracy = accuracy_score(y_test,y_pred) * 100

    return accuracy_train, accuracy_test, precision, recall, f1,accuracy
