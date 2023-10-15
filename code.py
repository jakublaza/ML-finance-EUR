import numpy as np
import pandas as pd



import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.exceptions
import sklearn.decomposition
import sklearn.metrics
import sklearn.compose
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer, accuracy_score, RocCurveDisplay, roc_curve
from sklearn.model_selection import GridSearchCV


# Load the train set
train = pd.read_csv('train.csv')
target = train.iloc[:, 0]
data = train.iloc[:, 1:]


# Train-test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data, target, test_size=0.2,  random_state=42, stratify=target)




scale_val = np.sum(target == 0 )/np.sum(target == 1)

# Final Pipeline
transformer = sklearn.preprocessing.MinMaxScaler()
polynomial = sklearn.preprocessing.PolynomialFeatures(2, include_bias=False, interaction_only=False)
pca = sklearn.decomposition.PCA(0.9, whiten=True)
model = xgb.XGBClassifier(objective = 'binary:logistic',
                           n_estimators = 2500, max_depth = 12, eta = 0.01, gamma = 1, reg_lambda = 2, subsample = 0.9, scale_pos_weight = 1.5) 

pipeline = sklearn.pipeline.Pipeline([ ('transformet', transformer), ('polynomial', polynomial), ('pca', pca),   ('model', model)]) 

model = pipeline.fit(X_train, y_train)


predictions = model.predict(X_test)
acc = sklearn.metrics.accuracy_score(y_test, predictions)
print(acc)


# Load test dataset for final predictions
test_final = pd.read_csv('test (1).csv')

predictions = model.predict(test_final)

# save to txt file
bitstring_array = predictions.astype(int)
bitstring = ''.join(str(bit) for bit in bitstring_array)

file_path = 'predictions.txt'

with open(file_path, 'w') as file:
    file.write(bitstring)



##############
## First CV ##
##############

# CV = sklearn.model_selection.GridSearchCV(pipeline, { 'model__n_estimators':[ 10, 100, 1000, 2000, 5000],'model__max_depth':[3, 6, 9, 12, 20] }, 
#                                      refit = True)
# model_CV = CV.fit(X_train, y_train) 

# print(model_CV.best_params_)
# print(model_CV.best_score_)

# predictions = model_CV.predict(X_test)
# acc = sklearn.metrics.accuracy_score(y_test, predictions)
# print(acc)



###################
## randomized CV ##
###################

# hyperparameter_grid = {
#     'model__n_estimators': [2000, 2500, 3000],
#     'model__max_depth': [12, 20, 25],
#     'model__learning_rate': [0.01, 0.1, 0.5], 
#     'model__gamma': [0, 1, 2, 3, 4], 
#     'model__reg_lambda' : [0, 1, 2, 3, 4],
#     'model__subsample' : [0.85, 0.9],
#     'model__scale_pos_weight' : [1.5, scale_val, 5]

# }

# folds = 3
# skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)
# random_cv = sklearn.model_selection.RandomizedSearchCV(estimator=pipeline,
#             param_distributions=hyperparameter_grid,
#             cv=3, 
#             n_iter=5,
#             scoring = 'accuracy',
#             n_jobs = 5, # to run in parrarel
#             verbose = 1, # get info while running
#             return_train_score = True,
#             random_state=42)

# model_rcv = random_cv.fit(X_train, y_train) 

# print(model_rcv.best_params_)
# print(model_rcv.best_score_)

# predictions = model_rcv.predict(X_test)
# acc = sklearn.metrics.accuracy_score(y_test, predictions)
# print(acc)









