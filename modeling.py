import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier

CROSS_VAL_CV = 5

def decision_tree(train_data: pd.DataFrame, train_target: pd.Series, test_data: pd.DataFrame, test_target: pd.Series):
    tree = DecisionTreeClassifier()
    tree.fit(train_data, train_target)
    test_prediction = tree.predict(test_data)
    accuracy_metrics(test_target, test_prediction)
    return tree


def accuracy_metrics(actual: np.array, predict: np.array):
    print(classification_report(actual,predict))
    print(accuracy_score(actual,predict))
    pass