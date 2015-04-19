import csv

sub = open(r'submission.csv')

rows = []
for t, row in enumerate(sub):
    row = row.strip().split(',')
    if t >= 1:
        for i in range(1, len(row)):
            row[i] = float(row[i])
        maxPredict = max(row[1:])
        maxId = row.index(maxPredict)
        if maxPredict > 0.95:
            for i in range(1, len(row)):
                row[i] = 0.0
            row[maxId] = 1.0
    rows.append(row)
    
predictions_file = open("c_submission.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerows(rows)
predictions_file.close()
