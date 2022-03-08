import joblib
import pandas as pd
from joblib import dump
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

best_study = joblib.load("optuna/study_LGB.pkl")
df: pd.DataFrame = joblib.load("data/fraud_dataset_transformed.pkl")
X = df.drop(columns=['rating'], axis=1)
y = df['rating'].apply(lambda el: 0 if el == 'OK' else 1)

# impute numerical data with mean values
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)
X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, y, test_size=0.30, random_state=42)

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

reg = LGBMClassifier(boosting_type=boostingtype,
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
clf = RFE(reg, step=0.02, n_features_to_select=trial["n_features"])
pipeline = Pipeline([
    ('rfe_feature_selection', clf),
    ('clf', clf)
])

train = X_TRAIN
test = Y_TRAIN

pipeline.fit(train, test)
dump(pipeline, 'lightgbm_model.joblib')
