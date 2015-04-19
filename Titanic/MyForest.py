import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import decomposition
from sklearn import cross_validation

train_df = pd.read_csv(r'imputeTrain.csv', header=0)
test_df = pd.read_csv('imputeTest.csv', header=0)

train_df.Fare[train_df['Fare']<10]=10
train_df.SibSp[train_df['SibSp']>=5]=3
train_df.SibSp[(train_df['SibSp']==3) | (train_df['SibSp']==4)]=2
train_df.SibSp[(train_df['SibSp']==0) | (train_df['SibSp']==1) | (train_df['SibSp']==2)]=1
train_df.Age[train_df['Age']<18]=1
train_df.Age[(train_df['Age']<40)&(train_df['Age']>=18)]=2
train_df.Age[(train_df['Age']<60)&(train_df['Age']>=40)]=3
train_df.Age[train_df['Age']>=60]=4

test_df.Fare[test_df['Fare']<10]=10
test_df.SibSp[test_df['SibSp']>=5]=3
test_df.SibSp[(test_df['SibSp']==3) | (test_df['SibSp']==4)]=2
test_df.SibSp[(test_df['SibSp']==0) | (test_df['SibSp']==1) | (test_df['SibSp']==2)]=1
test_df.Age[test_df['Age']<18]=1
test_df.Age[(test_df['Age']<40)&(test_df['Age']>=18)]=2
test_df.Age[(test_df['Age']<60)&(test_df['Age']>=40)]=3
test_df.Age[test_df['Age']>=60]=4
##train_df['Fare']=train_df['Fare']**0.25
##test_df['Fare']=train_df['Fare']**0.25

##train_df.Fare[train_df['Fare']>=39]=39
##train_df.Fare[(train_df['Fare']<39)&(train_df['Fare']>=30)]=3
##train_df.Fare[(train_df['Fare']<30)&(train_df['Fare']>=20)]=2
##train_df.Fare[(train_df['Fare']<20)&(train_df['Fare']>=10)]=1
##train_df.Fare[train_df['Fare']<10]=0

##test_df.Fare[test_df['Fare']>=39]=4
##test_df.Fare[(test_df['Fare']<39)&(test_df['Fare']>=30)]=3
##test_df.Fare[(test_df['Fare']<30)&(test_df['Fare']>=20)]=2
##test_df.Fare[(test_df['Fare']<20)&(test_df['Fare']>=10)]=1
##test_df.Fare[test_df['Fare']<10]=0


ids = test_df['PassengerId'].values
sur = train_df['Survived'].values
train_df = train_df.drop(['PassengerId','Survived','Parch'], axis=1)
test_df = test_df.drop(['PassengerId','Parch'], axis=1) 
train_data = train_df.values
test_data = test_df.values


##scaler = preprocessing.StandardScaler().fit(train_data)
##train_data = preprocessing.scale(train_data)
##scaler2 = preprocessing.StandardScaler().fit(test_data)
##test_data = preprocessing.scale(test_data)


##pca = decomposition.PCA(n_components=3)
##pca.fit(train_data)
##train_data=pca.transform(train_data)
##test_pca=pca.transform(test_scaled)


##X_train, X_test, y_train, y_test = cross_validation.train_test_split(train_data, sur, test_size=0.3, random_state=0)


print 'Training...'
forest = RandomForestClassifier(n_estimators=200)
forest = forest.fit( train_data, sur )

print 'Predicting...'
##output = forest.score(X_test, y_test)
output = forest.predict(test_data).astype(int)

##ÊäﾳöÖÁExcelÖÐ
predictions_file = open("myforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'
