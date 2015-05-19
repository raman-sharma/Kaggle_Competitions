import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
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
a2 = np.mat(device_label).T
device_enc = np.array(enc_device.fit_transform(a2).todense())

y = train_df['outcome'].values
train_df = train_df.drop(['bidder_id', 'outcome', 'max_merch',
                          'max_device', 'max_country', 'max_ip', 'max_url',
                          'account', 'address'], axis=1)
train_data = train_df.values
train = np.hstack((train_data, device_enc[:len(train_data)]))

Ids = test_df['bidder_id'].values
test_df = test_df.drop(['bidder_id', 'max_merch',
                        'max_device', 'max_country', 'max_ip', 'max_url',
                        'account', 'address'], axis=1)
test_data = test_df.values
test = np.hstack((test_data, device_enc[len(train_data):]))

scl = StandardScaler()
train = scl.fit_transform(train)
test = scl.transform(test)


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


def testAlgorithm(X_train, X_test, y_train, y_test):
    forest = RandomForestClassifier(n_estimators=2000, random_state=123,
                 max_features=26, class_weight='auto')
    forest = forest.fit(X_train, y_train)
    proba1 = forest.predict_proba(X_test)
    proba = []
    for t, row in enumerate(proba1):
        proba.append(row[1])
    proba = np.array(proba)
    y_test = np.array(y_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, proba, pos_label=1)
    loss = metrics.auc(fpr, tpr)
    print loss
    return loss


def myCV(data=train, y=y, trials=10):
    skf = cross_validation.StratifiedKFold(y, n_folds=10)
    error = 0.0
    for train_index, test_index in skf:
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = y[train_index], y[test_index]
        error += testAlgorithm(X_train, X_test, y_train, y_test)
    return error/trials

score = myCV()

print score

##forest = RandomForestClassifier(n_estimators = 1000, max_features='auto',
##                                random_state = 123, class_weight='auto')
##forest = forest.fit(train, y)
##proba = forest.predict_proba(test)
##predictions_file = open("submissions/submission_rf4.csv", "wb")
##open_file_object = csv.writer(predictions_file)
##open_file_object.writerow(['bidder_id', 'prediction'])
##for t, row in enumerate(Ids):
##    s = []
##    s.append(row)
##    s.append(proba[t][1])
##    open_file_object.writerow(s)
##predictions_file.close()
##print 'Done.'
