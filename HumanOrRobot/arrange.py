import csv
from csv import DictReader

features = ['bidder_id', 'bid_num', 'auc_device', 'auc_country', 'auc_ip', 'auc_url',
            'auc_num', 'max_merch', 'merch_num', 'max_device', 'device_num', 'max_country',
            'country_num', 'max_ip', 'ip_num', 'max_url', 'url_num']

filename = "sorted_bidder.csv"
open_file = open("arrange.csv", "wb")
open_file_object = csv.writer(open_file)
open_file_object.writerow(features)
writable = dict()
last_row = dict()
write_row = list()
write_rows = list()
count = 0

def update_writable(writable, row):
    writable.setdefault('num', 0)
    writable.setdefault('auction', dict())
    writable.setdefault('merchandise', dict())
    writable.setdefault('device', dict())
    writable.setdefault('country', dict())
    writable.setdefault('ip', dict())
    writable.setdefault('url', dict())
    writable['num'] += 1
    # Update each auction the bidder has particapated
    writable['auction'].setdefault(row['auction'], dict())
    writable['auction'][row['auction']].setdefault('device', dict())
    writable['auction'][row['auction']].setdefault('country', dict())
    writable['auction'][row['auction']].setdefault('ip', dict())
    writable['auction'][row['auction']].setdefault('url', dict())
    writable['auction'][row['auction']]['device'].setdefault(row['device'], 0)
    writable['auction'][row['auction']]['country'].setdefault(row['country'], 0)
    writable['auction'][row['auction']]['ip'].setdefault(row['ip'], 0)
    writable['auction'][row['auction']]['url'].setdefault(row['url'], 0)
    writable['auction'][row['auction']]['device'][row['device']] += 1
    writable['auction'][row['auction']]['country'][row['country']] += 1
    writable['auction'][row['auction']]['ip'][row['ip']] += 1
    writable['auction'][row['auction']]['url'][row['url']] += 1
    # Update other features of the bidder
    writable['merchandise'].setdefault(row['merchandise'], 0)
    writable['merchandise'][row['merchandise']] += 1
    writable['device'].setdefault(row['device'], 0)
    writable['device'][row['device']] += 1
    writable['country'].setdefault(row['country'], 0)
    writable['country'][row['country']] += 1
    writable['ip'].setdefault(row['ip'], 0)
    writable['ip'][row['ip']] += 1
    writable['url'].setdefault(row['url'], 0)
    writable['url'][row['url']] += 1

def argmax(d):
    """
    :param
    d: A dictionary
    :return:
    max_class: the key that have the maximum value of the dictionary
    max_value: the maximum value of the dictionary
    """
    max_value = 0
    max_class = None
    for key in d:
        if d[key] > max_value:
            max_value = d[key]
            max_class = key
    return max_value, max_class

def max_auction(d, target):
    """
    :param
    d: A auction dictionary
    target: 'country', 'device', 'ip' or 'url'
    :return:
    max_value: the maximum number of the target
    """
    max_value = 0
    for key in d:
        current_max = len(d[key][target].keys())
        if current_max > max_value:
            max_value = current_max
    return max_value

for t, row in enumerate(DictReader(open(filename))):

    if t == 0:
        last_row = row
        update_writable(writable, row)
        continue

    if row['bidder_id'] != last_row['bidder_id']:
        write_row = [last_row['bidder_id'], writable['num'], max_auction(writable['auction'], 'device'),
                     max_auction(writable['auction'], 'country'), max_auction(writable['auction'], 'ip'),
                     max_auction(writable['auction'], 'url'), len(writable['auction'].keys()),
                     argmax(writable['merchandise'])[1], len(writable['merchandise'].keys()), argmax(writable['device'])[1],
                     len(writable['device'].keys()), argmax(writable['country'])[1],
                     len(writable['country'].keys()), argmax(writable['ip'])[1], len(writable['ip'].keys()),
                     argmax(writable['url'])[1], len(writable['url'].keys())]
        write_rows.append(write_row)
        count += 1
        writable = dict()
        write_row = list()

    update_writable(writable, row)

    if (count%1000 == 0) and (count != 0):
        print t, count

    last_row = row

open_file_object.writerows(write_rows)
open_file.close()
