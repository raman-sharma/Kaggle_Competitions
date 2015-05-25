import pandas as pd
import numpy as np
import csv

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier, BaggingClassifier)
from sklearn.svm import SVC
from sklearn import cross_validation
from random import random, randint
from math import log, exp, sqrt,factorial
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


enc_device = OneHotEncoder()
lab_device = LabelEncoder()

train_df = pd.read_csv(r'pre_train.csv', header = 0)
test_df = pd.read_csv(r'pre_test.csv', header = 0)

device_train = train_df['max_device'].values
device_test = test_df['max_device'].values
device = np.hstack((device_train, device_test))
device_label = lab_device.fit_transform(device)
device_enc = enc_device.fit_transform(np.mat(device_label).T).toarray()

y = train_df['outcome'].values
train_df = train_df.drop(['bidder_id', 'outcome', 'max_merch',
                          'max_device', 'max_country', 'max_ip', 'max_url',
                          'account', 'address'], axis=1)
train_data = train_df.values
train = np.hstack((train_data, device_enc[:len(train_data)]))

idx = test_df['bidder_id'].values
test_df = test_df.drop(['bidder_id', 'max_merch',
                        'max_device', 'max_country', 'max_ip', 'max_url',
                        'account', 'address'], axis=1)
test_data = test_df.values
test = np.hstack((test_data, device_enc[len(train_data):]))

scl = StandardScaler()
train = scl.fit_transform(np.log(train+2))
test = scl.transform(np.log(test+2))


def train_and_test(X_train, X_test, y_train, y_test):
    forest = BaggingClassifier(n_estimators=500, random_state=1234)
    forest = forest.fit(X_train, y_train)
    proba = forest.predict_proba(X_test)
    proba = proba[:, 1]
    y_test = np.array(y_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, proba, pos_label=1)
    loss = metrics.auc(fpr, tpr)
    print loss
    return loss


def kfold_validation(data=train, y=y, trials=10):
    skf = cross_validation.StratifiedKFold(y, n_folds=10)
    error = 0.0
    for train_index, test_index in skf:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = y[train_index], y[test_index]
        error += train_and_test(X_train, X_test, y_train, y_test)
    return error/trials


score = kfold_validation()
print score

forest = BaggingClassifier(n_estimators=1000, random_state=1234)
forest = forest.fit(train, y)
proba = forest.predict_proba(test)
proba = proba[:, 1]
submission = pd.DataFrame({"bidder_id": idx, "prediction": proba})
submission.to_csv("submissions/submission_bag1.csv", index=False)
print 'Done.'
