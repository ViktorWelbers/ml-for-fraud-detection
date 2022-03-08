import joblib
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import balanced_accuracy_score, precision_score, accuracy_score, recall_score

from parameter_tuning import pr_auc as aURPC

X_train = joblib.load('data/x_train.pkl')
y_train = joblib.load('data/y_train.pkl')
X_test = joblib.load('data/x_test.pkl')
y_test = joblib.load('data/y_test.pkl')

model = joblib.load('models/lightgbm_model.joblib')

y_pred = model.predict(X_test)

test_AURPC = aURPC(y_test, y_pred)
test_balanced_acc = balanced_accuracy_score(y_test, y_pred)
test_precision_score = precision_score(y_test, y_pred)
test_accuracy_score = accuracy_score(y_test, y_pred)
test_recall_score = recall_score(y_test, y_pred)

print(f'AURPC = {test_AURPC}')
print(f'Balanced ACC = {test_balanced_acc}')
print(
    f'Precision = {test_precision_score}  The precision is intuitively'
    f' the ability of the classifier not to label as positive a sample that is negative.')
print(f'Accuracy = {test_accuracy_score}')
print(f'Recall = {test_recall_score} The ability of the classifier to find all the positive sample')

ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, display_labels=['OK', 'FRAUD'])
plt.show()

result = permutation_importance(model, X_train, y_train, n_repeats=5)

sorted_idx = result.importances_mean.argsort()
plt.figure()
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.tick_params(axis='both', which='major', labelsize=15)
ax.boxplot(result.importances[sorted_idx].T,
           vert=False, labels=X_train.columns[sorted_idx])
ax.set_title("Permutation Importance", pad=20)
fig.tight_layout()
plt.show()
