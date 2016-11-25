from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.cluster import KMeans,MeanShift


__logistic = LogisticRegression()
__my_svm = svm.SVC()



def logistic_regression_train(data,labels):
    __logistic.fit(data, labels)


def logistic_regression_predict(test_data):
    return __logistic.predict(test_data)


def svm_train(data,labels):
    __my_svm.fit(data,labels)


def svm_predict(test_data):
    return __my_svm.predict(test_data)


def cluster_meanShift(data):

    my_meanShift = MeanShift(bandwidth=0.5,bin_seeding=True,n_jobs=-1)

    my_meanShift.fit(data)
    cluster_prediction = my_meanShift.predict(data)


    return cluster_prediction


def cluster_kmeans(data,k):
    ##KMeans
    my_kmean = KMeans(n_clusters=k,init='k-means++',n_jobs=-1)
    cluster_prediction = my_kmean.fit_predict(data)
    return cluster_prediction