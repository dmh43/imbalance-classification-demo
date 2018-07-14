import numpy as np
import numpy.linalg as la
import pandas as pd
import numpy.random as rn
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

from imblearn.over_sampling import SMOTE, RandomOverSampler

def create_imbalance(X, y):
  assert len(X) == len(y)
  num_to_keep = int(0.1 * (y == 1).sum())
  true_indexes_to_keep = rn.choice(np.where(y == 1)[0], (num_to_keep,), replace=False)
  indexes_to_keep = np.concatenate((np.where(y == 0)[0],
                                    true_indexes_to_keep))
  indexes_to_drop = np.array(list(set(np.where(y == 1)[0]) - set(indexes_to_keep)))
  return X[indexes_to_keep], y[indexes_to_keep], X[indexes_to_drop], y[indexes_to_drop]

def main():
  X, y = load_breast_cancer(True)
  cms = {}
  accs = {}
  num_trials = 100
  for trial_num in range(num_trials):
    if trial_num % 10 == 0: print(trial_num)
    X_train_raw, X_test, y_train_raw, y_test = train_test_split(X, y, test_size=0.2)
    X_train, y_train, X_dropped, y_dropped = create_imbalance(X_train_raw, y_train_raw)
    for to_assess in [{'name': 'l2', 'model': LogisticRegressionCV(penalty='l2')},
                      {'name': 'l2 reweighted', 'model': LogisticRegressionCV(penalty='l2', class_weight='balanced')},
                      {'name': 'SMOTE', 'model': LogisticRegressionCV(penalty='l2'), 'options': 'SMOTE'},
                      {'name': 'SMOTE reweighted', 'model': LogisticRegressionCV(penalty='l2', class_weight='balanced'), 'options': 'SMOTE'},
                      {'name': 'random sampling', 'model': LogisticRegressionCV(penalty='l2'), 'options': 'random_oversampling'},
                      {'name': 'random sampling balanced', 'model': LogisticRegressionCV(penalty='l2', class_weight='balanced'), 'options': 'random_oversampling'}]:
      X_train_sampled = X_train
      y_train_sampled = y_train
      model = to_assess['model']
      if to_assess.get('options') == 'SMOTE':
        X_train_sampled, y_train_sampled = SMOTE().fit_sample(X_train, y_train)
      elif to_assess.get('options') == 'random_oversampling':
        X_train_sampled, y_train_sampled = RandomOverSampler().fit_sample(X_train, y_train)
      model.fit(X_train_sampled, y_train_sampled)
      predictions = model.predict(np.concatenate((X_test, X_dropped), 0))
      if trial_num == 0:
        cms[to_assess['name']] = confusion_matrix(np.concatenate((y_test, y_dropped), 0), predictions)
        accs[to_assess['name']] = accuracy_score(np.concatenate((y_test, y_dropped), 0), predictions)
      else:
        cms[to_assess['name']] += confusion_matrix(np.concatenate((y_test, y_dropped), 0), predictions)
        accs[to_assess['name']] += accuracy_score(np.concatenate((y_test, y_dropped), 0), predictions)
  print(to_assess['name'])
  [print(cm/num_trials) for cm in cms.values()]
  [print(acc/num_trials)for acc in accs.values()]
  return


if __name__ == "__main__": main()
