from csv import DictReader
import csv

features = ['id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain',\
                               'app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15',\
                               'C16','C17','C18','C19','C20','C21']
features2 = ['id','click','hour','C1','banner_pos','site_id','site_domain','site_category','app_id','app_domain',\
                               'app_category','device_id','device_ip','device_model','device_type','device_conn_type','C14','C15',\
                               'C16','C17','C18','C19','C20','C21','ip_num']

def feat_dict(fname):
    
    w = {}
    
    for t, row in enumerate(DictReader(open(fname))):
        
        if t == 0:
            for key in row:
                if key == 'device_ip':
                    w[key] = {}
                
        for key in row:
            if key == 'device_ip':
                value = row[key]
                if value not in w[key]:
                    w[key][value] = 0
                w[key][value] += 1

        if t % 10000 == 0:
            print t

    return w

def rare_replace(fname, outname):

    f = open(outname, 'wb')
    open_file_object = csv.writer(f)
    open_file_object.writerow(features2)
    
    for t, row in enumerate(DictReader(open(fname))):

        if 'click' not in row:
            row['click'] = 0

        value = row['device_ip']
        row['ip_num'] = '_GOOD_'
        if value not in w['device_ip'] or w['device_ip'][value] <= 20:
                row['ip_num'] = '_RARE_'


        open_file_object.writerow([row[feat] for feat in features2])

        if t % 10000 == 0:
            print t
    
    f.close()


w = feat_dict(r'C:\kaggle\CTR\train.csv')
rare_replace(r'C:\kaggle\CTR\train.csv', r'C:\kaggle\CTR\pre_train.csv')
rare_replace(r'C:\kaggle\CTR\test.csv', r'C:\kaggle\CTR\pre_test.csv')


