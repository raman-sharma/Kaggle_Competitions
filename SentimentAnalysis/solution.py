import csv
import numpy as np
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn import cross_validation
from sklearn import linear_model
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import grid_search
from sklearn.ensemble import GradientBoostingClassifier

def crossValidation(parameters):
    kf = cross_validation.KFold(156060, n_folds=10)
    score=0
    j=0
    for train_index, test_index in kf:
        X_train, X_test = X1[train_index], X1[test_index]
##        X_train2, X_test2 = X2[train_index], X2[test_index]
        y_train, y_test = sentiment[train_index], sentiment[test_index]
        Xt1,Xt2=Xt[train_index], Xt[test_index]
        clf1 = linear_model.SGDClassifier(loss=parameters[0],penalty=parameters[1],fit_intercept=parameters[2],\
                                                           n_iter=parameters[3],l1_ratio=0.05)
        clf1 = clf1.fit(X_train, y_train)
        output1 = clf1.predict(X_train).astype(int)
        output2 = clf1.predict(X_test).astype(int)
        XX1=[]
        XX2=[]
        for i in range(len(Xt1)):
            a=np.hstack((Xt1[i],output1[i]))
            XX1.append(a)
        for i in range(len(Xt2)):
            b=np.hstack((Xt2[i],output2[i]))
            XX2.append(b)
        XX1=np.array(XX1)
        XX2=np.array(XX2)
        forest=GradientBoostingClassifier()
        forest=forest.fit(XX1,y_train)
        score+=forest.score(XX2,y_test)
##        nb = MultinomialNB()
##        nb = nb.fit(X_train, y_train)
##        output += nb.score(X_test,y_test)
##        clf2 = linear_model.SGDClassifier(loss=parameters2[0],penalty=parameters2[1],fit_intercept=parameters2[2],\
##                                                           n_iter=parameters2[3],l1_ratio=0.05)
##        clf2 = clf2.fit(X_train2, y_train)
##        output3 = clf2.predict(X_test2).astype(int)
##        output += clf.score(X_test, y_test)
        j+=1
        print j,score
    score=score/10
    return score

def features(document):
    document=word_tokenize(document)
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains(%s)' % word] = (word in document_words)
    return features

def read_train(t):
    phraseID=[]
    sentenceID=[]
    sentence=[]
    sentiment=[]
    for i in t[1:]:
        tokens = word_tokenize(i)
        s=i[i.find('\t',i.find('\t')+1)+1:i.rfind('\t',i.rfind('\t'))]
        phraseID.append(tokens[0])
        sentenceID.append(tokens[1])
        sentence.append(s)
        sentiment.append(int(tokens[-1]))
    sentiment=np.array(sentiment)
    return phraseID,sentenceID,sentence,sentiment

def read_test(t):
    phraseID=[]
    sentenceID=[]
    sentence=[]
    for i in t[1:]:
        tokens = word_tokenize(i)
        s=i[i.find('\t',i.find('\t')+1)+1:i.find('\n')]
        phraseID.append(tokens[0])
        sentenceID.append(tokens[1])
        sentence.append(s)
    return phraseID,sentenceID,sentence

def tuning():
    p_dict = {'loss':('hinge','log','modified_huber'), 'penalty':('l2','elasticnet'),'fit_intercept':(True,False),'n_iter':(3,5,7)}
    output=0
    parameters=[(a,b,c,d) for a in p_dict['loss'] for b in p_dict['penalty'] for c in p_dict['fit_intercept'] for d in p_dict['n_iter']]
    for i in parameters:
        score=crossValidation(i)
        if score>=output:
            output=score
            p=i
    return output,p

def syns(s):
    result = []
    tokens=word_tokenize(s)
    for word in tokens:
        ss = nltk.wordnet.wordnet.synsets(word)
        result.extend(str(s) for s in ss if ".n." not in str(s))
    return " ".join(result)

def transform(sentence):
    return [syns(s) for s in sentence]

##train_df = pd.read_table(r'train.tsv', header=0)

##all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
##word_features = all_words.keys()[:100]

train = open(r'train.tsv')
test = open(r'test.tsv')

t=[]
for row in train:
    t.append(row)
t2=[]
for row in test:
    t2.append(row)

phraseID,sentenceID,sentence,sentiment=read_train(t)
phraseID2,sentenceID2,sentence2=read_test(t2)
##syn_sentence=transform(sentence)

####Bag of Words
vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,3),min_df=1)
vectorizer1 = vectorizer.fit(sentence)
##vectorizer2 = vectorizer.fit(syn_sentence)
X1 = vectorizer.transform(sentence)
##X2 = vectorizer.transform(syn_sentence)
##Xtest = vectorizer.transform(sentence2)
ch2 = SelectKBest(chi2, k=20)
Xt = ch2.fit_transform(X1, sentiment)
Xt=Xt.toarray()


####Build Classifiers
##clf = linear_model.SGDClassifier(loss='modified_huber', penalty='elasticnet', n_iter=7)
##clf = clf.fit(X, sentiment)
##output = clf.predict(Xtest).astype(int)


####Implement the cross validation function
parameters=['modified_huber','elasticnet',True,7]
score=crossValidation(parameters)
print score


####Implement the grid search function
##score,parameters=tuning()
##print score,parameters

##j=0
##for i in range(len(output)):
##    if output[i]==sentiment[i]:
##        j+=1
##print float(j)/len(output)

##Naive Bayes
##clf = MultinomialNB()
##clf.fit(Xt, sentiment)
##output = clf.predict(Xt).astype(int)

##def ent(s):
##    tokens = word_tokenize(s)
##    tagged = nltk.pos_tag(tokens)
##    entities = nltk.chunk.ne_chunk(tagged)
##    return entities


####Output the submission data
##predictions_file = open("submission.csv", "wb")
##open_file_object = csv.writer(predictions_file)
##open_file_object.writerow(["PhraseId","Sentiment"])
##open_file_object.writerows(zip(phraseID2, output))
##predictions_file.close()
