import csv
from csv import DictReader

sample = open("sampleSubmission.csv")
submission = "submissions/submission_rf4.csv"
Ids = dict()

for row in DictReader(sample):
    Ids.setdefault(row['bidder_id'], 1)

for row in DictReader(open(submission)):
    del Ids[row['bidder_id']]

predictions_file = open("submissions/submission_comp_rf4.csv", "wb")
open_file_object = csv.writer(predictions_file)

for row in open(submission):
    row = row.strip().split(',')
    open_file_object.writerow(row)

for key in Ids:
    row = [key, 0.0]
    open_file_object.writerow(row)

predictions_file.close()
