from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cluster import KMeans,MeanShift


import database

__logistic = LogisticRegression()
__my_svm = svm.SVC()



def logistic_regression_train(data,labels,save_model):
    __logistic.fit(data, labels)

    if save_model:
        database.save_classifier_model(__logistic,'')


def logistic_regression_predict(test_data):
    return __logistic.predict(test_data), __logistic.decision_function(test_data)


def svm_train(data,labels):
    __my_svm.fit(data,labels)


def svm_predict(test_data):
    return __my_svm.predict(test_data)


def cluster_meanShift(data,save_model):

    my_meanShift = MeanShift(bandwidth=0.5,bin_seeding=True,n_jobs=-1)

    my_meanShift.fit(data)

    #cluster_prediction = my_meanShift.predict(data)

    if save_model:
        database.save_classifier_model(my_meanShift,'')



def cluster_kmeans(data,k):
    ##KMeans
    my_kmean = KMeans(n_clusters=k,init='k-means++',n_jobs=-1)
    cluster_prediction = my_kmean.fit_predict(data)
    return cluster_prediction


def model_Freezing(nSteps, nVars, RNN, lrnRate, pDrop=0.5):

    from keras.models import Model
    from keras.layers import Input, Dense, LSTM
    from keras.optimizers import Nadam


    if RNN[1]:
        model_in = Input(batch_shape=(RNN[2], nSteps, nVars))
    else:
        model_in = Input(shape=(nSteps, nVars))

    x = LSTM(units=RNN[0],
             implementation=0,
             return_sequences=True,
             stateful=RNN[1],
             unroll=True,
             dropout=pDrop,
             recurrent_dropout=pDrop,
             activation='tanh')(model_in)

    x = LSTM(units=RNN[0],
             implementation=0,
             return_sequences=False,
             stateful=RNN[1],
             unroll=True,
             dropout=pDrop,
             recurrent_dropout=pDrop,
             activation='tanh')(x)

    x = Dense(units=RNN[0],
              activation='tanh')(x)

    x = Dense(units=2,
              activation='softmax')(x)

    model_out = x


    model = Model(inputs=model_in, outputs=model_out)

    model.compile(loss='binary_crossentropy',    # binary_crossentropy, categorical
                  optimizer=Nadam(lr=lrnRate),   # Nadam, Adam, RMSprop, SGD
                  sample_weight_mode='None')

    model.summary(line_length=100)


    return model