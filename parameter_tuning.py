import joblib
import optuna
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.feature_selection import RFE
from sklearn.metrics import precision_recall_curve, auc, make_scorer
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline


def pr_auc(y_true, probas_pred)-> float:
    # calculate precision-recall curve
    p, r, _ = precision_recall_curve(y_true, probas_pred)
    # calculate area under curve
    return auc(r, p)

def objective(trial):
    boostingtype = 'gbdt'
    min_child_samples = trial.suggest_categorical('min_child_samples',[5,10, 20, 30, 40, 50, 60 , 70, 80, 90, 100])
    num_leaves = trial.suggest_int('num_leaves', 2, 256)
    # learning rate
    learning_rate = trial.suggest_float("eta", 1e-8, 1.0, log=True)
    # Amount of trees build
    n_estimators = trial.suggest_int('n_estimators', 2, 300)
    # L2 regularization weight.
    reg_lambda = trial.suggest_float("lambda", 1e-8, 1.0, log=True)
    # L1 regularization weight.
    reg_alpha = trial.suggest_float("alpha", 1e-8, 1.0, log=True)
    # maximum depth of the tree, signifies complexity of the tree.
    max_depth = trial.suggest_int("max_depth", -1, 25, step=2)
    # define loss function

    objective = 'binary'

    params = {"max_bin": trial.suggest_categorical('max_bin', [63, 127, 255, 511])}

    n_features = trial.suggest_categorical("n_features", [0.5, 0.6, 0.7, 0.8, 0.9, 1])
    subsample = trial.suggest_float("subsample", 0.2, 1.0)
    subsample_freq = trial.suggest_int("bagging_freq", 0, 7)
    colsample_bytree = trial.suggest_float("colsample_bytree", 0.2, 1.0)


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

    rfe = RFE(clf, n_features_to_select=n_features, step=0.02)
    pipe = make_pipeline(rfe, clf)
    cv_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
    cv = cross_val_score(pipe, X_train, y_train, n_jobs=5, cv=cv_folds, scoring=metric)
    trial.set_user_attr('crossval scores', cv)
    return cv.mean()


if __name__ == '__main__':
    df: pd.DataFrame = joblib.load("data/fraud_dataset_transformed.pkl")
    X = df.drop(columns=['rating'], axis=1)
    y = df['rating'].apply(lambda el: 0 if el == 'OK' else 1)

    #impute numerical data with mean values
    X = X.apply(lambda x: x.fillna(x.mean()),axis=0)

    #conduct traintest split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42)

    #Create Metric for unbalanced class dist
    metric = make_scorer(pr_auc, needs_proba=True)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, timeout=600)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    joblib.dump(study, "optuna/study_LGB.pkl")
    joblib.dump(trial, "optuna/best_trial_LGB.pkl")
