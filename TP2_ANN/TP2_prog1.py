

from sklearn.datasets import fetch_openml
from sklearn import model_selection
import numpy as np
from sklearn import neural_network
from sklearn import metrics

mnist = fetch_openml('mnist_784')

percent = 0.7

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(mnist.data, mnist.target, train_size=percent)


clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,))
clf.fit(xtrain, ytrain)
ypred = clf.predict(xtest)

precision = metrics.precision_score(ytest, ypred, average='micro')
print("Precision :" + str(precision))

print("Classe de l'image 4 : " + str(ytest[4]))
print("Classe prédite de l'image 4 : " + str(clf.predict(xtest)[4]))

tup = ();
for k in range(1,101,10):
    tup += (50)
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(mnist.data, mnist.target, train_size=percent)

    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup)
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)

    precision = metrics.precision_score(ytest, ypred, average='micro')
    print("Precision pour %d couches de 50 neurones :" %(k,precision))


