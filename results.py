import joblib
from sklearn.model_selection import train_test_split

best_study = joblib.load("optuna/study_LGB.pkl")
df: pd.DataFrame = joblib.load("data/fraud_dataset_transformed.pkl")
X = df.drop(columns=['rating'], axis=1)
y = df['rating'].apply(lambda el: 0 if el == 'OK' else 1)

# impute numerical data with mean values
X = X.apply(lambda x: x.fillna(x.mean()), axis=0)

X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(X, y, test_size=0.30, random_state=42)

model = joblib.load('lightgbm_model.joblib')

pred = model.predict(X_TEST)

