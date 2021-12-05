import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score
import pickle

np.random.seed(123)
CROSS_VAL_CV = 5


def accuracy_metrics(actual: np.array, predict: np.array):
    print(classification_report(actual,predict))
    print(accuracy_score(actual,predict))
    pass

def decision_tree(train_data: pd.DataFrame, train_target: pd.Series, test_data: pd.DataFrame, test_target: pd.Series):
    tree = DecisionTreeClassifier()
    clf = GridSearchCV(tree, {
        'criterion': ("gini", "entropy")
    })
    clf.fit(train_data, train_target)
    test_prediction = clf.predict(test_data)
    accuracy_metrics(test_target, test_prediction)
    return clf

def random_forest(train_data: pd.DataFrame, train_target: pd.Series, test_data: pd.DataFrame, test_target: pd.Series):
    rmf = GridSearchCV(
        RandomForestClassifier(),
        {
            'n_estimators': list(range(100, 1100, 100))
            ,'max_depth': [None, 5, 10, 15, 20]
        }
    )
    rmf.fit(train_data, train_target)
    test_prediction = rmf.predict(test_data)
    accuracy_metrics(test_target, test_prediction)
    return rmf

def logistic_regression(train_data: pd.DataFrame, train_target: pd.Series, test_data: pd.DataFrame, test_target: pd.Series):
    lgr = GridSearchCV(
        LogisticRegression(),
        {
            'penalty': ('l1', 'l2', 'elasticnet', 'none')
        }
    )
    lgr.fit(train_data, train_target)
    test_prediction = lgr.predict(test_data)
    accuracy_metrics(test_target, test_prediction)
    return lgr

def support_vector(train_data: pd.DataFrame, train_target: pd.Series, test_data: pd.DataFrame, test_target: pd.Series):
    svc = GridSearchCV(
        SVC(),
        {
            'C': list(np.arange(0.1, 1, 0.1)) + list(range(1, 10, 1)) + list(10, 100, 10),
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
        }
    )
    svc.fit(train_data, train_target)
    test_prediction = svc.predict(test_data)
    accuracy_metrics(test_target, test_prediction)
    return svc

def main(data: pd.DataFrame, target_column_name: str):
    target = data[target_column_name]
    features = data.drop(target_column_name, axis=1)
    train_data, train_target, test_data, test_target = train_test_split(features, target, train_size=0.8)
    print('=='*20)
    print('Decision Tree')
    dtree = decision_tree(train_data, train_target, test_data, test_target)
    print('=='*20)
    print('Random Forest')
    rmf = random_forest(train_data, train_target, test_data, test_target)
    print('=='*20)
    print('Logistic Regression')
    lgr = logistic_regression(train_data, train_target, test_data, test_target)
    print('=='*20)
    print('Support Vector Machine')
    svc = support_vector(train_data, train_target, test_data, test_target)
    return {
        'DecisionTree':dtree
        ,'RandomForest':rmf
        ,'logisticRegression':lgr
        ,'SupportVectorMachine':svc
    }

if __name__ == '__main__':
    cleanData = pd.read_csv('cleanDF.csv')
    models = main(cleanData)
    for model in models.keys():
        pickle.dump(models[model], f'{model}.sav')
