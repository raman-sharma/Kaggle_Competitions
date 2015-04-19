import pandas as pd
import csv

features = ['feat_' + str(i) for i in range(1, 94)]

train_df = pd.read_csv(r'train.csv', header = 0)
test_df = pd.read_csv(r'test.csv', header = 0)

Ids = test_df['id'].values
test_df = test_df.drop(['id'], axis=1)
test_data = test_df.values

sums = []
for i in test_data:
    s = i.sum()
    sums.append(s)

predictions_file = open("pre_test.csv", "wb")
open_file_object = csv.writer(predictions_file)
head = ['id']
head.extend(features)
head.append('sum')
open_file_object.writerow(head)
for t, row in enumerate(Ids):
    s = []
    s.append(row)
    s.extend(test_data[t])
    s.append(sums[t])
    open_file_object.writerow(s)
predictions_file.close()
