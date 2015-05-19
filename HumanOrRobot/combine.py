import csv
import pandas as pd

sub_xgb1 = pd.read_csv(r'submissions/submission_comp.csv', header = 0)
sub_xgb2 = pd.read_csv(r'submissions/submission_comp2.csv', header = 0)
sub_xgb3 = pd.read_csv(r'submissions/submission_comp3.csv', header = 0)
sub_xgb4 = pd.read_csv(r'submissions/submission_comp4.csv', header = 0)
sub_xgb5 = pd.read_csv(r'submissions/submission_comp5.csv', header = 0)
sub_nn1 = pd.read_csv(r'submissions/submission_comp_nn1.csv', header = 0)
sub_nn2 = pd.read_csv(r'submissions/submission_comp_nn2.csv', header = 0)
sub_nn3 = pd.read_csv(r'submissions/submission_comp_nn3.csv', header = 0)
sub_nn4 = pd.read_csv(r'submissions/submission_comp_nn4.csv', header = 0)
sub_rf1 = pd.read_csv(r'submissions/submission_comp_rf1.csv', header = 0)
sub_rf2 = pd.read_csv(r'submissions/submission_comp_rf2.csv', header = 0)
sub_rf3 = pd.read_csv(r'submissions/submission_comp_rf3.csv', header = 0)
sub_rf4 = pd.read_csv(r'submissions/submission_comp_rf4.csv', header = 0)

Ids = sub_xgb1['bidder_id'].values
sub_xgb1 = sub_xgb1.drop(['bidder_id'], axis=1)
sub_xgb2 = sub_xgb2.drop(['bidder_id'], axis=1)
sub_xgb3 = sub_xgb3.drop(['bidder_id'], axis=1)
sub_xgb4 = sub_xgb4.drop(['bidder_id'], axis=1)
sub_xgb5 = sub_xgb5.drop(['bidder_id'], axis=1)
sub_nn1 = sub_nn1.drop(['bidder_id'], axis=1)
sub_nn2 = sub_nn2.drop(['bidder_id'], axis=1)
sub_nn3 = sub_nn3.drop(['bidder_id'], axis=1)
sub_nn4 = sub_nn4.drop(['bidder_id'], axis=1)
sub_rf1 = sub_rf1.drop(['bidder_id'], axis=1)
sub_rf2 = sub_rf2.drop(['bidder_id'], axis=1)
sub_rf3 = sub_rf3.drop(['bidder_id'], axis=1)
sub_rf4 = sub_rf4.drop(['bidder_id'], axis=1)

data_xgb1 = sub_xgb1.values
data_xgb2 = sub_xgb2.values
data_xgb3 = sub_xgb3.values
data_xgb4 = sub_xgb4.values
data_xgb5 = sub_xgb5.values
data_nn1 = sub_nn1.values
data_nn2 = sub_nn2.values
data_nn3 = sub_nn3.values
data_nn4 = sub_nn4.values
data_rf1 = sub_rf1.values
data_rf2 = sub_rf2.values
data_rf3 = sub_rf3.values
data_rf4 = sub_rf4.values

comb = []
for t, i in enumerate(data_xgb1):
    xgb = i*0.2 + data_xgb2[t]*0.0 + data_xgb3[t]*0.2 + data_xgb4[t]*0.3 + data_xgb5[t]*0.3
    nn = data_nn1[t]*0.5 + data_nn2[t]*0.5
    rf = data_rf1[t]*0.1 + data_rf2[t]*0.1 + data_rf3[t]*0.2 + data_rf4[t]*0.6
    c = 0.3 * xgb + 0.2 * nn + 0.5 * rf
    comb.append(c)

predictions_file = open("submissions/combination.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['bidder_id', 'prediction'])
for t, row in enumerate(Ids):
    s = []
    s.append(row)
    s.append(comb[t][0])
    open_file_object.writerow(s)
predictions_file.close()
