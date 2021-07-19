import optuna
from pkg.imbalanced.imblearn.over_sampling._smote.base import MySMOTE

def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    smote = MySMOTE(sampling_strategy="auto", random_state=0, my_smote=x, weight=np.array(X.std().array))

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    mod = DecisionTreeClassifier(max_depth=3, random_state=42)
    mod.fit(X_train_resampled, y_train_resampled)

    y_pred = mod.predict(X_val)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    from sklearn.metrics import roc_curve, auc
    precision, recall, thresholds = roc_curve(y_val, y_pred)
    score = auc(precision, recall)
    
    print(str(x) + " then " + str(score))
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

trial = study.best_trial

print('AUC: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))