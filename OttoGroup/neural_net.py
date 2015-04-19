import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from lasagne.layers import DenseLayer
from lasagne.layers import InputLayer
from lasagne.layers import DropoutLayer
from lasagne import layers
from lasagne.nonlinearities import softmax
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet

def load_train_data(path):
    df = pd.read_csv(path)
    X = df.values.copy()
    np.random.shuffle(X)
    X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).astype(np.int32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder, scaler

def load_test_data(path, scaler):
    df = pd.read_csv(path)
    X = df.values.copy()
    X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
    X = scaler.transform(X)
    return X, ids

def make_submission(clf, X_test, ids, encoder, name='my_neural_net_submission.csv'):
    y_prob = clf.predict_proba(X_test)
    with open(name, 'w') as f:
        f.write('id,')
        f.write(','.join(encoder.classes_))
        f.write('\n')
        for id, probs in zip(ids, y_prob):
            probas = ','.join([id] + map(str, probs.tolist()))
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(name))

X, y, encoder, scaler = load_train_data('pre_train.csv')
X_test, ids = load_test_data('pre_test.csv', scaler)
num_classes = len(encoder.classes_)
num_features = X.shape[1]

layers0 = [('input', InputLayer),
           ('dense0', DenseLayer),
           ('dropout', DropoutLayer),
           ('dense1', DenseLayer),
           ('dropout1', DropoutLayer),
           ('dense2', DenseLayer),
           ('output', DenseLayer)]

net0 = NeuralNet(layers=layers0,
                 
                 input_shape=(None, num_features),
                 dense0_num_units=300,
                 dropout_p=0.5,
                 dense1_num_units=500,
                 dropout1_p=0.5,
                 dense2_num_units=300,
                 output_num_units=num_classes,
                 output_nonlinearity=softmax,
                 
                 update=nesterov_momentum,
                 update_learning_rate=0.01,
                 update_momentum=0.9,
                 
                 verbose=1,
                 max_epochs=200)

net2 = NeuralNet(layers=[('input', layers.InputLayer),
                         ('conv1', layers.Conv2DLayer),
                         ('pool1', layers.MaxPool2DLayer),
                         ('conv2', layers.Conv2DLayer),
                         ('pool2', layers.MaxPool2DLayer),
                         ('conv3', layers.Conv2DLayer),
                         ('pool3', layers.MaxPool2DLayer),
                         ('hidden4', layers.DenseLayer),
                         ('hidden5', layers.DenseLayer),
                         ('output', layers.DenseLayer),],
                 input_shape=(None, 1, 1, num_features),
                 conv1_num_filters=32, conv1_filter_size=(1, 3), pool1_ds=(1, 2),
                 conv2_num_filters=64, conv2_filter_size=(1, 2), pool2_ds=(1, 2),
                 conv3_num_filters=128, conv3_filter_size=(1, 2), pool3_ds=(1, 2),
                 hidden4_num_units=500, hidden5_num_units=500,
                 output_num_units=num_classes, output_nonlinearity=softmax,

                 update_learning_rate=0.01,
                 update_momentum=0.9,

                 max_epochs=100,
                 verbose=1,
                 )

##X = X.reshape(-1, 1, 1, num_features)
##X_test = X_test.reshape(-1, 1, 1, num_features)
net0.fit(X, y)
make_submission(net0, X_test, ids, encoder)
