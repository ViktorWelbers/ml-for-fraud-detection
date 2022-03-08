import joblib
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline

best_study = joblib.load("../optuna/study_LGB.pkl")

# impute numerical data with mean values

X_TRAIN = joblib.load('../data/x_train.pkl')
y_TRAIN = joblib.load('../data/y_train.pkl')

trial = best_study.best_params

boostingtype = "gbdt"
min_child_samples = trial['min_child_samples']
num_leaves = trial['num_leaves']
# learning rate
learning_rate = trial["eta"]
# Amount of trees build
n_estimators = trial['n_estimators']
# L2 regularization weight.
reg_lambda = trial["lambda"]
# L1 regularization weight.
reg_alpha = trial["alpha"]
# maximum depth of the tree, signifies complexity of the tree.
max_depth = trial["max_depth"]
# define loss function
objective = 'binary'
params = {"max_bin": trial['max_bin']}

# Bagging
subsample = trial["subsample"]
subsample_freq = trial["bagging_freq"]
colsample_bytree = trial["colsample_bytree"]

clf = LGBMClassifier(boosting_type=boostingtype,
                     num_leaves=num_leaves,
                     learning_rate=learning_rate,
                     n_estimators=n_estimators,
                     min_child_samples=min_child_samples,
                     max_depth=max_depth,
                     colsample_bytree=colsample_bytree,
                     subsample=subsample,
                     reg_alpha=reg_alpha,
                     reg_lambda=reg_lambda,
                     objective=objective,
                     subsample_freq=subsample_freq,
                     **params
                     )

# Recursive Feature elimination in pipeline
rec = RFE(clf, step=0.02, n_features_to_select=trial["n_features"])
pipeline = Pipeline([
    ('rfe_feature_selection', rec),
    ('clf', clf)
])

pipeline.fit(X_TRAIN, y_TRAIN)
dump(pipeline, '../models/lightgbm_model.joblib')
