import csv
import sys
sys.path.append('/home/yejiming/xgboost/wrapper')
import xgboost as xgb
import pandas as pd

train_df = pd.read_csv(r'pre_train.csv', header = 0)
test_df = pd.read_csv(r'pre_test.csv', header = 0)

y = train_df['outcome'].values
train_df = train_df.drop(['bidder_id', 'outcome', 'max_merch',
                          'max_device', 'max_country', 'max_ip', 'max_url',
                          'account', 'address'], axis=1)
train_data = train_df.values

Ids = test_df['bidder_id'].values
test_df = test_df.drop(['bidder_id', 'max_merch',
                        'max_device', 'max_country', 'max_ip', 'max_url',
                        'account', 'address'], axis=1)
test_data = test_df.values

dtrain = xgb.DMatrix(train_data, label = y)
dtest = xgb.DMatrix(test_data)

param = {'objective':'binary:logistic', 'nthread':8,
         'eval_metric':'auc', 'bst:max_depth':20, 'gamma':0,
         'bst:min_child_weight':3, 'bst:subsample':1.0,
         'bst:colsample_bytree':1.0, 'bst:eta':0.01}

num_round = 300
print 'Training...'
bst = xgb.cv(param, dtrain, num_round, nfold = 5)
##bst = xgb.train(param, dtrain, num_round)
##
##print 'Predicting...'
##preds = bst.predict(dtest)
##
##predictions_file = open("submissions/submission3.csv", "wb")
##open_file_object = csv.writer(predictions_file)
##open_file_object.writerow(['bidder_id', 'prediction'])
##for t, row in enumerate(Ids):
##    s = []
##    s.append(row)
##    s.append(preds[t])
##    open_file_object.writerow(s)
##predictions_file.close()
##print 'Done.'
