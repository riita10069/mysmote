## Without Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

X = df.iloc[:, 0:2]
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)

mod = DecisionTreeClassifier(max_depth=3, random_state=42)
mod.fit(X_train, y_train)

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
