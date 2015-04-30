import csv
from csv import DictReader

test_features = ['bidder_id', 'bid_num', 'max_auc', 'auc_num', 'max_merch', 'merch_num',
                 'max_device', 'device_num', 'max_country', 'country_num', 'max_ip',
                 'ip_num', 'max_url', 'url_num', 'account', 'address']
train_features = test_features.copy()
train_features.append('outcome')

train = open('train.csv')
test = open('test.csv')
arrange = open('arrange.csv')
train_bidder = dict{}
test_bidder = dict{}

pre_train = open("pre_train.csv", "wb")
pre_test = open("pre_test.csv", "wb")
open_file_train = csv.writer(pre_train)
open_file_test = csv.writer(pre_test)
open_file_train.writerow(train_features)
open_file_test.writerow(test_features)

for row in DictReader(train):
    train_bidder[row['bidder_id']]['account'] = row['payment_account']
    train_bidder[row['bidder_id']]['address'] = row['address']
    train_bidder[row['bidder_id']]['outcome'] = row['outcome']

for row in DictReader(test):
    test_bidder[row['bidder_id']]['account'] = row['payment_account']
    test_bidder[row['bidder_id']]['address'] = row['address']


for row in arrange:
    row = row.strip().split(',')

    if row[0] in train_bidder:
        row.append(train_bidder[row[0]]['account'])
        row.append(train_bidder[row[0]]['address'])
        row.append(train_bidder[row[0]]['outcome'])
        open_file_train.writerow(row)

    elif row[0] in test_bidder:
        row.append(test_bidder[row[0]]['account'])
        row.append(train_bidder[row[0]]['address'])
        open_file_test.writerow(row)

    else:
        print row[0]

pre_train.close()
pre_test.close()
