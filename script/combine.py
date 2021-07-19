# Combine
from imblearn.combine import SMOTETomek
smt = SMOTETomek(random_state=42)

X_train_resampled, y_train_resampled = smt.fit_resample(X, y)
resampled = X_train_resampled.join(y_train_resampled)
sns.scatterplot(x='x', y='y', hue='Class', data=resampled)

mod = DecisionTreeClassifier(max_depth=3, random_state=42)
mod.fit(X_train_resampled, y_train_resampled)

y_pred = mod.predict(X_test)

print('Confusion matrix(test):\n{}'.format(confusion_matrix(y_test, y_pred)))
print('Accuracy(test) : %.5f' %accuracy_score(y_test, y_pred))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('precision : %.4f'%(tp / (tp + fp)))
print('recall : %.4f'%(tp / (tp + fn)))

from sklearn.metrics import roc_cu
rve, auc
precision, recall, thresholds = roc_curve(y_test, y_pred)
score = auc(precision, recall)
print('AUC : %.5f' %score)
