import csv
from csv import DictReader
import numpy as np

features = ['bidder_id', 'bid_num', 'avg_auction_times', 'avg_auction', 'auc_device', 'auc_country', 'auc_ip', 'auc_url',
            'auc_device_max', 'auc_country_max', 'auc_ip_max', 'auc_url_max',
            'auc_device_min', 'auc_country_min', 'auc_ip_min', 'auc_url_min',
            'auc_num', 'max_merch', 'merch_num', 'max_device', 'device_num', 'max_country',
            'country_num', 'max_ip', 'ip_num', 'max_url', 'url_num', 'avg_times', 'min_times', 'max_times',
            'min_amount', 'avg_amount', 'max_amount', 'success_num']

filename = "final.csv"
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
    writable.setdefault('time', dict())
    writable['num'] += 1
    # Update each auction the bidder has particapated
    writable['auction'].setdefault(row['auction'], dict())
    writable['auction'][row['auction']].setdefault('device', dict())
    writable['auction'][row['auction']].setdefault('country', dict())
    writable['auction'][row['auction']].setdefault('ip', dict())
    writable['auction'][row['auction']].setdefault('url', dict())
    writable['auction'][row['auction']].setdefault('time', dict())
    writable['auction'][row['auction']].setdefault('amount', 0)
    writable['auction'][row['auction']].setdefault('success', 0)
    writable['auction'][row['auction']]['device'].setdefault(row['device'], 0)
    writable['auction'][row['auction']]['country'].setdefault(row['country'], 0)
    writable['auction'][row['auction']]['ip'].setdefault(row['ip'], 0)
    writable['auction'][row['auction']]['url'].setdefault(row['url'], 0)
    writable['auction'][row['auction']]['time'].setdefault(row['time'], 0)
    writable['auction'][row['auction']]['device'][row['device']] += 1
    writable['auction'][row['auction']]['country'][row['country']] += 1
    writable['auction'][row['auction']]['ip'][row['ip']] += 1
    writable['auction'][row['auction']]['url'][row['url']] += 1
    writable['auction'][row['auction']]['time'][row['time']] += 1
    if row['amount'] > writable['auction'][row['auction']]['amount']:
        writable['auction'][row['auction']]['amount'] = row['amount']
    if int(row['success']) == 1:
        writable['auction'][row['auction']]['success'] = 1
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
    writable['time'].setdefault(row['time'], 0)
    writable['time'][row['time']] += 1


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


def avg_auction(d, target):
    """
    :param
    d: An auction dictionary
    target: 'country', 'device', 'ip' or 'url'
    :return:
    avg_value: the average value of the target
    """
    sum_value = 0
    count = 0
    for key in d:
        current_value = len(d[key][target].keys())
        sum_value += current_value
        count += 1
    return float(sum_value) / count


def max_auction(d, target):
    """
    :param
    d: An auction dictionary
    target: 'country', 'device', 'ip' or 'url'
    :return:
    max_value: the maximum value of the target
    """
    max_value = 0
    for key in d:
        current_value = len(d[key][target].keys())
        if current_value > max_value:
            max_value = current_value
    return max_value


def min_auction(d, target):
    """
    :param
    d: An auction dictionary
    target: 'country', 'device', 'ip' or 'url'
    :return:
    min_value: the minimum value of the target
    """
    min_value = 9999
    for key in d:
        current_value = len(d[key][target].keys())
        if current_value < min_value:
            min_value = current_value
    return min_value


def avg_auction_time(d):
    """
    :param
    d: An auction dictionary
    :return:
    avg_time: the average time the bidder takes to make a bid
    """
    sum_times = 0
    count = 0
    for key in d:
        single_times = sum(d[key]['time'].values())
        sum_times += single_times
        count += 1
    return float(sum_times) / count


def avg_auction_time2(d):
    """
    :param
    d: An auction dictionary
    :return:
    avg_time: the average time the bidder takes to make a bid
    """
    sum_times = 0
    count = 0
    for key in d:
        single_times = np.array(d[key]['time'].values()).mean()
        sum_times += single_times
        count += 1
    return float(sum_times) / count


def auction_amount(d):
    """
    :param
    d: An auction dictionary
    :return:
    avg_amount: the average amount the bidder takes to make a bid
    max_amount: the maximum amount the bidder takes to make a bid
    min_amount: the minimum amount the bidder takes to make a bid
    """
    sum_amount = 0
    max_amount = 0
    min_amount = 99999999
    count = 0
    for key in d:
        amount = int(d[key]['amount'])
        sum_amount += amount
        count += 1
        if amount < min_amount:
            min_amount = amount
        if amount > max_amount:
            max_amount = amount
    return min_amount, float(sum_amount) / count, max_amount


def auction_success(d):
    """
    :param
    d: An auction dictionary
    :return:
    success_num: number of successful bid
    """
    num_success = 0
    for key in d:
        num_success += int(d[key]['success'])
    return num_success


for t, row in enumerate(DictReader(open(filename))):

    if t == 0:
        last_row = row
        update_writable(writable, row)
        continue

    if row['bidder_id'] != last_row['bidder_id']:
        if np.array(writable['time'].values()).mean() > 1:
            avg_times = 1
        else:
            avg_times = 0
        if np.array(writable['time'].values()).max() > 1:
            max_times = 1
        else:
            max_times = 0
        min_amount, avg_amount, max_amount = auction_amount(writable['auction'])
        write_row = [last_row['bidder_id'], writable['num'], avg_auction_time2(writable['auction']),
                     avg_auction_time(writable['auction']), avg_auction(writable['auction'], 'device'),
                     avg_auction(writable['auction'], 'country'), avg_auction(writable['auction'], 'ip'),
                     avg_auction(writable['auction'], 'url'), max_auction(writable['auction'], 'device'),
                     max_auction(writable['auction'], 'country'), max_auction(writable['auction'], 'ip'),
                     max_auction(writable['auction'], 'url'), min_auction(writable['auction'], 'device'),
                     min_auction(writable['auction'], 'country'), min_auction(writable['auction'], 'ip'),
                     min_auction(writable['auction'], 'url'), len(writable['auction'].keys()),
                     argmax(writable['merchandise'])[1], len(writable['merchandise'].keys()), argmax(writable['device'])[1],
                     len(writable['device'].keys()), argmax(writable['country'])[1],
                     len(writable['country'].keys()), argmax(writable['ip'])[1], len(writable['ip'].keys()),
                     argmax(writable['url'])[1], len(writable['url'].keys()), avg_times,
                     np.array(writable['time'].values()).min(), max_times, min_amount, avg_amount, max_amount,
                     auction_success(writable['auction'])]
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
