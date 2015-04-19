import pandas as pd
import numpy as np
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import ensemble, feature_extraction, preprocessing
from sklearn.ensemble import AdaBoostClassifier

train_df = pd.read_csv(r'train.csv', header = 0)
test_df = pd.read_csv(r'test.csv', header = 0)

y = train_df['target'].values
train_df = train_df.drop(['target', 'id'], axis=1)
train_data = train_df.values

Ids = test_df['id'].values
test_df = test_df.drop(['id'], axis=1)
test_data = test_df.values

##tfidf = feature_extraction.text.TfidfTransformer()
##train_data = tfidf.fit_transform(train_data).toarray()
##test_data = tfidf.transform(test_data).toarray()

print 'Training...'
forest = GradientBoostingClassifier(n_estimators=200, verbose=1,
                                    learning_rate = 0.2, max_depth=3)
forest2 = RandomForestClassifier(n_estimators = 400, verbose = 1,
                                 max_features = 13)
learner = AdaBoostClassifier(base_estimator = forest2, n_estimators = 50)
forest = forest.fit(train_data, y)
learner = learner.fit(train_data, y)

print 'Predicting...'
output1 = forest.predict_proba(test_data)
output2 = learner.predict_proba(test_data)

output = []
for t, row in enumerate(output1):
    tmp = np.vstack([output1[t], output2[t]])
    tmp = np.average(tmp, axis = 0)
    output.append(tmp)
output = np.array(output)

predictions_file = open("submission.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(['Id', 'Class_1', 'Class_2', 'Class_3',
                           'Class_4', 'Class_5', 'Class_6',
                           'Class_7', 'Class_8', 'Class_9'])

for t, row in enumerate(Ids):
    s = []
    s.append(row)
    s.extend(output[t])
    open_file_object.writerow(s)
predictions_file.close()
print 'Done.'
