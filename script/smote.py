# Smote
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.6, random_state=0)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
mod = DecisionTreeClassifier(max_depth=3, random_state=42)
mod.fit(X_train_resampled, y_train_resampled)
y_pred = mod.predict(X_test)

print('Confusion matrix(test):\n{}'.format(confusion_matrix(y_test, y_pred)))
print('Accuracy(test) : %.5f' %accuracy_score(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('precision : %.4f'%(tp / (tp + fp)))
print('recall : %.4f'%(tp / (tp + fn)))

from sklearn.metrics import roc_curve, auc
precision, recall, thresholds = roc_curve(y_test, y_pred)
score = auc(precision, recall)
print('AUC : %.5f' %score)
