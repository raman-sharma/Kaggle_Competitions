import pandas as pd
import numpy as np
import csv as csv
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn import decomposition

train_df = pd.read_csv(r'train.csv', header=0)
test_df = pd.read_csv('test.csv', header=0)

df=train_df.append(test_df)
df.Cabin[df.Cabin.notnull()]=1
df.Cabin[df.Cabin.isnull()]=0

##Impute Age using Name
df.Name = df.Name.map( lambda x: x[x.find(',')+2:x.find('.')])
df.Name[(df.Name=='Don') | (df.Name=='Rev') | (df.Name=='Col') | (df.Name=='Capt') | (df.Name=='The Countess') | (df.Name=='Jonkheer') | (df.Name=='Dr') | (df.Name=='Major')]='Others'
df.Name[(df.Name=='Mme') | (df.Name=='Ms') | (df.Name=='Lady') | (df.Name=='Dona') | (df.Name=='Mrs')]='Lady'
df.Name[(df.Name=='Mlle') | (df.Name=='Miss')]='Miss'
df.Name[(df.Name=='Mr') | (df.Name=='Sir')]='Sir'

Names = list(enumerate(np.unique(df['Name'])))
Names_dict = { name : i for i, name in Names }
df.Name = df.Name.map( lambda x: Names_dict[x]).astype(int)

for i in df.Name.unique():
	if len(df.Age[( df.Name==i )& (df.Age.isnull()) ]) > 0:
		df.Age[( df.Name==i )& (df.Age.isnull()) ] = df.Age[df.Name==i].dropna().median()
		
# female = 0, Male = 1
df['Gender'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# All missing Embarked -> just make them embark from most common place
if len(df.Embarked[ df.Embarked.isnull() ]) > 0:
    df.Embarked[ df.Embarked.isnull() ] = df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
df.Embarked = df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

## Impute Fare by Pclass
for i in df.Pclass.unique():
        if len(df.Fare[ (df.Fare.isnull()) & df.Pclass==i ]) > 0:
                df.Fare[ df.Fare.isnull() ] = df.Fare[df.Pclass==i].dropna().median()

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
df = df.drop(['Sex', 'Ticket'], axis=1) 
train_df=df[0:891]
test_df=df[891:]
test_df = test_df.drop(['Survived'], axis=1)

df.to_csv('imputeFull.csv', index=False)
train_df.to_csv('imputeTrain.csv', index=False)
test_df.to_csv('imputeTest.csv', index=False)
