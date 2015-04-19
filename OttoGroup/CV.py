import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from random import random, randint
from math import log, exp, sqrt,factorial
from sklearn import preprocessing

train_df = pd.read_csv(r'train.csv', header = 0)
test_df = pd.read_csv(r'test.csv', header = 0)

y = train_df['target'].values
train_df = train_df.drop(['target', 'id'], axis=1)
train_data = train_df.values

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(y)

def multiclass_log_loss(y_true, y_pred, eps=1e-15):
    predictions = np.clip(y_pred, eps, 1 - eps)

    # normalize row sums to 1
    predictions /= predictions.sum(axis=1)[:, np.newaxis]

    actual = np.zeros(y_pred.shape)
    n_samples = actual.shape[0]
    actual[np.arange(n_samples), y_true.astype(int)] = 1
    vectsum = np.sum(actual * np.log(predictions))
    loss = -1.0 / n_samples * vectsum
    return loss

def divideData(data, test=0.1):
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for t, row in enumerate(data):
        if random() < test:
            X_test.append(row)
            y_test.append(y[t])
        else:
            X_train.append(row)
            y_train.append(y[t])
    return X_train, X_test, y_train, y_test

def testAlgorithm(X_train, X_test, y_train, y_test):
    forest = GradientBoostingClassifier(n_estimators = 200, verbose = 1,
                                        learning_rate = 0.2)
    forest2 = RandomForestClassifier(n_estimators = 400, verbose = 1,
                                     max_features = 13)
    learner = AdaBoostClassifier(base_estimator = forest2, n_estimators = 50)
    forest = forest.fit(X_train, y_train)
    learner = learner.fit(X_train, y_train)
    proba1 = forest.predict_proba(X_test)
    proba2 = learner.predict_proba(X_test)
    proba = []
    for t, row in enumerate(proba1):
        tmp = np.vstack([proba1[t], proba2[t]])
        tmp = np.average(tmp, axis = 0)
        proba.append(tmp)
    proba = np.array(proba)
    y_test = np.array(y_test)
    loss = multiclass_log_loss(y_test, proba, eps=1e-15)
    print loss
    return loss

def myCV(data=train_data, trials=10, test=0.1):
    error = 0.0
    for i in range(trials):
        X_train, X_test, y_train, y_test = divideData(data, test)
        error += testAlgorithm(X_train, X_test, y_train, y_test)
    return error/trials

score = myCV()

print score
