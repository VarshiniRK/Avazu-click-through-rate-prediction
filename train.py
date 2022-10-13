
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from google.colab import drive
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report, roc_auc_score, roc_curve, confusion_matrix, roc_auc_score, auc, log_loss
from imblearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline
import multiprocessing as mp
#from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
from statsmodels.stats.proportion import proportion_confint
from keras.wrappers.scikit_learn import KerasClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
import random
import gzip
from datetime import datetime
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
import pickle


warnings.filterwarnings("ignore")

types_train = {
    'id': np.dtype(int),
    'click': np.dtype(int),
    'hour': np.dtype(int),
    'C1': np.dtype(int),
    'banner_pos': np.dtype(int),
    'site_id': np.dtype(str),
    'site_domain': np.dtype(str), 
    'site_category': np.dtype(str),
    'app_id': np.dtype(str),
    'app_domain': np.dtype(str),
    'app_category': np.dtype(str),
    'device_id': np.dtype(str),
    'device_ip': np.dtype(str),
    'device_model': np.dtype(str),
    'device_type': np.dtype(int),
    'device_conn_type': np.dtype(int),
    'C14': np.dtype(int),
    'C15': np.dtype(int),
    'C16': np.dtype(int),
    'C17': np.dtype(int),
    'C18': np.dtype(int),
    'C19': np.dtype(int),
    'C20': np.dtype(int),
    'C21':np.dtype(int)
}

types_test = {
    'id': np.dtype(int),
    'hour': np.dtype(int),
    'C1': np.dtype(int),
    'banner_pos': np.dtype(int),
    'site_id': np.dtype(str),
    'site_domain': np.dtype(str), 
    'site_category': np.dtype(str),
    'app_id': np.dtype(str),
    'app_domain': np.dtype(str),
    'app_category': np.dtype(str),
    'device_id': np.dtype(str),
    'device_ip': np.dtype(str),
    'device_model': np.dtype(str),
    'device_type': np.dtype(int),
    'device_conn_type': np.dtype(int),
    'C14': np.dtype(int),
    'C15': np.dtype(int),
    'C16': np.dtype(int),
    'C17': np.dtype(int),
    'C18': np.dtype(int),
    'C19': np.dtype(int),
    'C20': np.dtype(int),
    'C21':np.dtype(int)
}

# Code to mount google drive in case you are loading the data from your google drive
from google.colab import drive
drive.mount('/gdrive')


n = 40428967  #total number of records in the clickstream data 
sample_size = 200000
skip_values = sorted(random.sample(range(1,n), n-sample_size)) 

parse_date = lambda val : datetime.strptime(val, '%y%m%d%H')

with gzip.open('/gdrive/My Drive/Diploma Project/avazu-ctr-prediction - Kaggle Dataset/train.gz') as f:
    df = pd.read_csv(f, parse_dates = ['hour'], date_parser = parse_date, dtype=types_train, skiprows = skip_values)

#Feature Engineering
df['hour_of_day'] = df["hour"].apply(lambda x: str(x.time())[:5])
#the feature hour_of_day only has hours to represent and not the minutes
df["hour_of_day"] = df["hour_of_day"].apply(lambda x: int(x.split(":")[0]))
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
df["day_of_week"] = df["hour"].apply(lambda x: days[x.weekday()])

# drop unwanted columns using subjective analysis and also drop highly correlated columns
cols = list(df.columns)
if any(col in cols for col in ["id", "hour","C17", "device_type"]):
  df = df.drop(["id", "hour","C17", "device_type"], axis=1)

def convert_obj_to_int(fm):
    
    object_list_columns = fm.columns
    object_list_dtypes = fm.dtypes
    print(object_list_columns)
    print(object_list_dtypes)
    for index in range(0,len(object_list_columns)):
        if object_list_dtypes[index] == object :
            fm[object_list_columns[index]] = fm[object_list_columns[index]].apply(lambda x: hash(x))
    return fm

df_hashed = convert_obj_to_int(df)
print(df_hashed.loc[0,:])
print(df_hashed.dtypes)

#ends

def create_train_valid_test_split(dF, test_percent, shuffle=True):

  if shuffle:
    dF = dF.sample(frac = 1).reset_index().drop("index", axis=1)

  cols = list(dF.columns)
  y = dF["click"].to_numpy()
  cols.remove('click')
  X = dF.loc[:, cols].to_numpy()
  print("Data shape before splitting: {}".format(X.shape))
  print("Labels shape before splitting: {}".format(y.shape))

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_percent, random_state=1)
  # X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.12, random_state=1)

  print("Training data shape: {}".format(X_train.shape))
  print("Training labels shapre: {}".format(y_train.shape))
  # print("Validation data shape: {}".format(X_valid.shape))
  # print("Validation labels shape: {}".format(y_valid.shape))
  print("Test data shape: {}".format(X_test.shape))
  print("Test labels shape: {}".format(y_test.shape))
  
  return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = create_train_valid_test_split(df_hashed, 0.10)

#iso = IsolationForest(contamination=0.1)
#yhat = iso.fit_predict(X_train)
#mask = yhat != -1
#X_iso, y_iso = X_train[mask, :], y_train[mask]

#model = DecisionTreeClassifier()
#over = SMOTE(random_state=2, sampling_strategy=0.4, k_neighbors=1)
#under = RandomUnderSampler(sampling_strategy=0.5)
#steps = [('o', over), ('u', under)]
#pipeline = Pipeline(steps=steps)
#Xn, yn = pipeline.fit_resample(X_iso, y_iso.ravel())
#cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
#scores = cross_val_score(model, Xn, yn, scoring='roc_auc', cv=cv, n_jobs=-1)
#score = np.mean(scores)
#print("k={}, Mean ROC AUC: {:.3f}".format(3, score))

#X_train = np.copy(Xn)
#y_train = np.copy(yn)

params = {}
params['boosting_type'] = 'gbdt'
params['objective'] = 'binary'
params['metric'] = 'binary_logloss'
# params['num_iterations'] = 100 # default
params['feature_fraction'] = 1.
params['bagging_fraction'] = 1.
params['nthreads'] = 8
# params['scale_pos_weight'] = 1 #positive_class_fraction
params['is_unbalance'] = False
params['max_bin'] = 2^12
params['n_estimators'] = 300
        
# parameter grid to use with cross-validation
param_grid = {}
param_grid['classifier__min_data_in_leaf'] = [30] 
param_grid['classifier__max_depth'] = [-1] 
param_grid['classifier__learning_rate'] = [0.03]
param_grid['classifier__min_data_per_group'] = [5]
param_grid['classifier__num_leaves'] = [100] # <= 2**max_depth
param_grid['classifier__regression_l2'] = [0.]
       

pipe = Pipeline([
    ('scale', StandardScaler()),
    ('fselect', SelectKBest(score_func=f_classif, k=15)),
    ('classifier',LGBMClassifier(**params))
])

model = GridSearchCV(pipe, param_grid=param_grid, cv=5, scoring='neg_log_loss')
print(X_train[0])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model accuracy: {}".format(accuracy))

import pickle
pickle.dump(model,open('pipeline.pkl','wb'))

# estimate log_loss
logloss = log_loss((y_test+1), probs)
print(logloss)

print('Best parameters set found on development set\n')
print(model.best_params_)

xgb_roc_auc = roc_auc_score(y_test, y_pred)
print(xgb_roc_auc)
