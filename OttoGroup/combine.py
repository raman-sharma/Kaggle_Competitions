import csv
import pandas as pd

sub1 = pd.read_csv(r'submission.csv', header = 0)
sub2 = pd.read_csv(r'submission1.csv', header = 0)
sub3 = pd.read_csv(r'submission2.csv', header = 0)
sub4 = pd.read_csv(r'submission3.csv', header = 0)

Ids = sub1['Id'].values
sub1 = sub1.drop(['Id'], axis=1)
sub2 = sub2.drop(['id'], axis=1)
sub3 = sub3.drop(['id'], axis=1)
sub4 = sub4.drop(['id'], axis=1)

data1 = sub1.values
data2 = sub2.values
data3 = sub3.values
data4 = sub4.values

comb = []
for t, i in enumerate(data1):
    c = i * 0.2 + data2[t] * 0.1 + data3[t] * 0.6 + data4[t] * 0.1
    comb.append(c)

predictions_file = open("combination.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['id', 'Class_1', 'Class_2', 'Class_3',
                           'Class_4', 'Class_5', 'Class_6',
                           'Class_7', 'Class_8', 'Class_9'])
for t, row in enumerate(Ids):
    s = []
    s.append(row)
    s.extend(comb[t])
    open_file_object.writerow(s)
predictions_file.close()
