import csv
import sys
sys.path.append('/home/yejiming/xgboost/wrapper')
import xgboost as xgb
import pandas as pd
from sklearn import preprocessing

train_df = pd.read_csv(r'pre_train.csv', header = 0)
test_df = pd.read_csv(r'pre_test.csv', header = 0)

y = train_df['target'].values
train_df = train_df.drop(['target', 'id'], axis=1)
train_data = train_df.values

lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(y)

Ids = test_df['id'].values
test_df = test_df.drop(['id'], axis=1)
test_data = test_df.values

dtrain = xgb.DMatrix(train_data, label = y)
dtest = xgb.DMatrix(test_data)

param = {'objective':'multi:softprob', 'num_class':9, 'nthread':8,
         'eval_metric':'mlogloss', 'bst:max_depth':30, 'bst:gamma':1,
         'bst:min_child_weight':4, 'bst:subsample':0.8,
         'bst:colsample_bytree':0.5, 'bst:eta':0.01}

num_round = 2600
print 'Training...'
bst = xgb.cv(param, dtrain, num_round, nfold = 5)
##bst = xgb.train(param, dtrain, num_round)
##
##print 'Predicting...'
##preds = bst.predict(dtest)
##
##predictions_file = open("submission2.csv", "wb")
##open_file_object = csv.writer(predictions_file)
##open_file_object.writerow(['id', 'Class_1', 'Class_2', 'Class_3',
##                           'Class_4', 'Class_5', 'Class_6',
##                           'Class_7', 'Class_8', 'Class_9'])
##for t, row in enumerate(Ids):
##    s = []
##    s.append(row)
##    s.extend(preds[t])
##    open_file_object.writerow(s)
##predictions_file.close()
##print 'Done.'
##    
