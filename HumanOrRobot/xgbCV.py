import csv
import sys
sys.path.append('/home/yejiming/xgboost/wrapper')
import xgboost as xgb
import pandas as pd
from sklearn import preprocessing

train_df = pd.read_csv(r'pre_train.csv', header = 0)
test_df = pd.read_csv(r'pre_test.csv', header = 0)

y = train_df['outcome']
train_df = train_df.drop(['bidder_id', 'outcome'], axis=1)
train_data = train_df.values

Ids = test_df['id'].values
test_df = test_df.drop(['id'], axis=1)
test_data = test_df.values

dtrain = xgb.DMatrix(train_data, label = y)
dtest = xgb.DMatrix(test_data)

param = {'objective':'multi:softprob', 'num_class':9, 'nthread':8,
         'eval_metric':'mlogloss', 'bst:max_depth':10, 'bst:gamma':1,
         'bst:min_child_weight':4, 'bst:subsample':0.8,
         'bst:colsample_bytree':0.5, 'bst:eta':0.1}

num_round = 200
print 'Training...'
bst = xgb.cv(param, dtrain, num_round, nfold = 5)
