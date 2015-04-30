import csv
from csv import DictReader

features = ['bidder_id', 'bid_num', 'max_auc', 'auc_num', 'max_merch', 'merch_num',
            'max_device', 'device_num', 'max_country', 'country_num', 'max_ip',
            'ip_num', 'max_url', 'url_num']

open_file = open("arrange.csv", "wb")
open_file_object = csv.writer(open_file)
open_file_object.writerow(features)
writable = dict()
last_row = dict()
write_row = list()
write_rows = list()
count = 0

def update_writable(writable, row):
    writable['num'] += 1
    writable['auction'].setdefault(row['auction'], 0)
    writable['auction'][row['auction']] += 1
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


for t, row in enumerate(DictReader(open(filename))):
    
    if t == 0:
        last_row = row
        update_writable(writable, row)
        continue

    if row['bidder_id'] == last_row['bidder_id']:
        update_writable(writable, row)

    else:
        write_row = [last_row['bidder_id'], writable['num'], argmax(writable['auction']),
                     len(writable['auction'].keys()), argmax(writable['merchandise']),
                     len(writable['merchandise'].keys()), argmax(writable['device']),
                     len(writable['device'].keys()), argmax(writable['country']),
                     len(writable['country'].keys()), argmax(writable['ip']), len(writable['ip'].keys()),
                     argmax(writable['url']), len(writable['url'].keys())]
        write_rows.append()
        count += 1
        writable = dict()
        write_row = list()
        update_writable(writable, row)

    if (count%1000 == 0) and (count != 0):
        open_file_object.writerows(write_rows) 
        print t
 
    last_row = row  

open_file.close() 
