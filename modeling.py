import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, plot_roc_curve
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import pickle

np.random.seed(123)
CROSS_VAL_CV = 5

def accuracy_metrics(actual: np.array, predict: np.array):
    print(classification_report(actual,predict))
    print(accuracy_score(actual,predict))
    print(confusion_matrix(actual,predict))
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
        RandomForestClassifier(n_jobs = -1),
        {
            'n_estimators': list(range(100, 600, 100))
        }
        ,verbose=5
    )
    rmf.fit(train_data, train_target)
    test_prediction = rmf.predict(test_data)
    accuracy_metrics(test_target, test_prediction)
    return rmf

def k_nearest_neighbour(train_data: pd.DataFrame, train_target: pd.Series, test_data: pd.DataFrame, test_target: pd.Series):
    knn = GridSearchCV(
        KNeighborsClassifier(),
        {
            'n_neighbors': list(range(5, 50, 5))
        },
        n_jobs=-1, verbose=5
    )
    knn.fit(train_data, train_target)
    test_prediction = knn.predict(test_data)
    accuracy_metrics(test_target, test_prediction)
    return knn

def feed_forward_network(train_data: pd.DataFrame, train_target: pd.Series, test_data: pd.DataFrame, test_target: pd.Series):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
    model = Sequential()
    model.add(Dense(16,input_shape=[13919],activation='relu'))
    model.add(Dense(8,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_data, train_target, epochs=20, batch_size=64)
    test_prediction = (model.predict(test_data) > 0.5).astype(int)
    accuracy_metrics(test_target, test_prediction)
    return model


def main(data: pd.DataFrame, target_column_name: str):
    target_column_name = 'target_column'
    target = data[target_column_name]
    features = data.drop(target_column_name, axis=1)
    train_data, test_data, train_target, test_target = train_test_split(features, target, train_size=0.8)
    print('=='*20)
    print('Decision Tree')
    dtree = decision_tree(train_data, train_target, test_data, test_target)
    print(dtree.best_params_)
    print(plot_roc_curve(dtree, test_data, test_target))
    pickle.dump(dtree, open('dtree.sav', 'wb'))
    print('=='*20)
    print('Random Forest')
    rmf = random_forest(train_data, train_target, test_data, test_target)
    print(rmf.best_params_)
    print(plot_roc_curve(rmf, test_data, test_target))
    pickle.dump(rmf, open('Randomforest.sav', 'wb'))
    print('=='*20)
    print('KNN')
    knn = k_nearest_neighbour(train_data, train_target, test_data, test_target)
    print(knn.best_params_)
    print(plot_roc_curve(knn, test_data, test_target))
    pickle.dump(knn, open('KNN.sav', 'wb'))
    print('=='*20)
    print('Feed Forward Network')
    nn = feed_forward_network(train_data, train_target, test_data, test_target)
    print(nn.best_params_)
    print(plot_roc_curve(nn, test_data, test_target))
    pickle.dump(nn, open('FeedForwardNetwork.sav', 'wb'))
