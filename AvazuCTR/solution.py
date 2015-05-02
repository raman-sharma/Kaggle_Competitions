from csv import DictReader
import csv
from math import exp, log, sqrt  

train = open(r'C:\kaggle\CTR\pre_train.csv') # Input the training data
test = open(r'C:\kaggle\CTR\pre_test.csv')   # Input the testing data
features = ['device_id','banner_pos','device_type','device_conn_type','device_model',\
            'site_id','app_id','site_domain','site_category','app_domain','app_category','device_ip']
features2 = ['device_id','banner_pos','device_type','device_conn_type','device_model',\
            'site_id','app_id','site_domain','site_category','app_domain','app_category']

site_features = ['device_id','banner_pos','device_type','device_conn_type','device_model','site_id',\
                 'site_domain','site_category','device_ip']
site_features2 = ['device_id','banner_pos','device_type','device_conn_type','device_model', 'site_id',\
                  'site_domain','site_category']

app_features = ['device_id','banner_pos','device_type','device_conn_type','device_model',\
            'app_id','app_domain','app_category','device_ip']
app_features2 = ['device_id','banner_pos','device_type','device_conn_type','device_model',\
            'app_id','app_domain','app_category']

hours = ['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15',
         '16','17','18','19','20','21','22','23']

alpha = 0.025  # learning rate
# for i in hours:
#     alpha[i] = 0.05
# alpha['01'] = 0.01
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 0.5     # L1 regularization, larger value means more regularized
L2 = 1.1     # L2 regularization, larger value means more regularized

D = 2 ** 24           # number of weights to use
interaction = False     # whether to enable poly2 feature interactions
holdafter = 29   # data after date N (exclusive) are used as validation

class ftrl_proximal(object):

    def __init__(self, alpha, beta, L1, L2, D, interaction):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = {}

    def _indices(self, x):

        # first yield index of the bias term
        yield 0

        # then yield the normal indices
        for index in x:
            yield index

        # now yield interactions (if applicable)
        if self.interaction:
            D = self.D
            L = len(x)

            x = sorted(x)
            for i in xrange(L):
                for j in xrange(i+1, L):
                    # one-hot encode interactions with hash trick
                    yield int(hash(str(x[i]) + '_' + str(x[j]))) % D

    def predict(self, x):

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = {}

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights
            # we are doing this at prediction instead of update time is because
            # this allows us for not storing the complete w
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        # cache the current w for update stage
        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def clean_parse_row(row):
    for k in features:
        yield (row[k], 1.0)
        for j in features:
            if k != j:
                yield (_make_interact([row[k],row[j]]), 1.0)

def _make_interact(iterable):
    return ':'.join(iterable)


def data(t):
    for t, row in enumerate(DictReader(t)):
        # process id
        ID = row['id']
        del row['id']

        # process clicks
        y = 0.
        if 'click' in row:
            if row['click'] == '1':
                y = 1.
            del row['click']

        # extract date
        date = int(row['hour'][4:6])
        # if date in [25,26]:
        #     row['day'] = 'weekend'
        # if date in [21,22,23,24,27,28,29,30,31]:
        #     row['day'] = 'weekday'

        # extract hour
        hour = row['hour'][6:]
        row['hour'] = hour

        # build 2-way and 3-way interactions
        if row['site_id'] != '85f751fd':
            type = 'site'
            del row['app_id']
            del row['app_domain']
            del row['app_category']
            if row['ip_num'] != '_RARE_':
                for k, i in enumerate(site_features):
                    for j in site_features[k+1:]:
                        row[i+':'+j] = _make_interact([row[i],row[j]])
                row['site_id:banner_pos:device_ip'] = _make_interact([row['site_id'], row['banner_pos'], row['device_ip']])

            if row['ip_num'] == '_RARE_':
                for k, i in enumerate(site_features2):
                    for j in site_features2[k+1:]:
                        row[i+':'+j] = _make_interact([row[i],row[j]])

        try:
            if row['app_id'] != 'ecad2386':
                type = 'app'
                del row['site_id']
                del row['site_domain']
                del row['site_category']
                if row['ip_num'] != '_RARE_':
                    for k, i in enumerate(app_features):
                        for j in app_features[k+1:]:
                            row[i+':'+j] = _make_interact([row[i],row[j]])
                    row['app_id:banner_pos:device_ip'] = _make_interact([row['app_id'], row['banner_pos'], row['device_ip']])

                if row['ip_num'] == '_RARE_':
                    for k, i in enumerate(app_features2):
                        for j in app_features2[k+1:]:
                            row[i+':'+j] = _make_interact([row[i],row[j]])

        except:
            pass

        # build x
        x = []
        for key in row:
            value = row[key]

            # one-hot encode everything with hash trick
            index = int(hash(key + '_' + value)) % D
            x.append(index)

        yield t,ID, y, x, date, hour, type


def logloss(p, y):
    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def training():

    loss = 0
    count = 0

    for t, ID, y, x, date, hour, type in data(train):

        if type == 'site':
            p = logit_site.predict(x)
        if type == 'app':
            p = logit_app.predict(x)
        loss += logloss(p, y)
        count += 1

        if t % 10000 == 0 and t > 0:
            print t, loss/count, hour, date

        if type == 'site':
            logit_site.update(x, p, y)
        if type == 'app':
            logit_app.update(x, p, y)

    print loss/count
    return loss/count

logit_site = ftrl_proximal(alpha, beta, L1, L2, D, interaction)
logit_app = ftrl_proximal(alpha, beta, L1, L2, D, interaction)

# logit_site = {}
# logit_app = {}
# for i in hours:
#     logit_site[i] = ftrl_proximal(alpha[i], beta, L1, L2, D, interaction)
#     logit_app[i] = ftrl_proximal(alpha[i], beta, L1, L2, D, interaction)

score = training()

print 'Done'

predictions_file = open("C:\kaggle\CTR\submission.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["id","click"])
for t, ID, y, x, date, hour, type in data(test):
    if type == 'site':
        p = logit_site.predict(x)
    if type == 'app':
        p = logit_app.predict(x)
    open_file_object.writerow([ID, p])
    if t % 10000 == 0:
        print t
predictions_file.close()
