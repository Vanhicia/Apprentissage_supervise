

from sklearn.datasets import fetch_openml
from sklearn import model_selection
import numpy as np
from sklearn import neural_network
from sklearn import metrics

mnist = fetch_openml('mnist_784')

index_vect = np.random.randint(70000, size=7000)
data = mnist.data[index_vect]
target = mnist.target[index_vect]

percent = 0.7

"""xtrain, xtest, ytrain, ytest = model_selection.train_test_split(mnist.data, mnist.target, train_size=percent)


clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,))
clf.fit(xtrain, ytrain)
ypred = clf.predict(xtest)

precision = metrics.precision_score(ytest, ypred, average='micro')
print("Precision :" + str(precision))

print("Classe de l'image 4 : " + str(ytest[4]))
print("Classe prédite de l'image 4 : " + str(clf.predict(xtest)[4]))

tup = ()
for k in range(1,101,10):
    tup += (50,) # TO DO : corriger tup !!!!
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)

    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup)
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)

    precision = metrics.precision_score(ytest, ypred, average='micro')
    print("Precision pour %d couches de 50 neurones : %f" %(k,precision))

# modèle 1 : 50 couches de 60 à 11 neurones
nb = 60
tup = ()
for i in range(0,50,1):
    tup += (nb-i,)

print("tup =")
print(tup)

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)

clf = neural_network.MLPClassifier(hidden_layer_sizes=tup)
clf.fit(xtrain, ytrain)
ypred = clf.predict(xtest)

precision = metrics.precision_score(ytest, ypred, average='micro')
print("Precision pour 50 couches de 60 à 11 neurones : %f" %(precision))

# modèle 2 : 20 couches de 60 à 11 neurones
nb = 60
tup = ()
for i in range(0,50,1):
    tup += (nb-i*3,)

print("tup =")
print(tup)

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)

clf = neural_network.MLPClassifier(hidden_layer_sizes=tup)
clf.fit(xtrain, ytrain)
ypred = clf.predict(xtest)

precision = metrics.precision_score(ytest, ypred, average='micro')
print("Precision pour 20 couches de 60 à 11 neurones : %f" %(precision))
"""
#TO DO : faire les modèles suivants

#50 couches de 50 neurones
tup = ()
for i in range(0,10,1):
    tup += (50,)

print("tup =")
print(tup)

for algo in ('lbfgs', 'sgd', 'adam'):
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)

    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup, solver=algo)
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)

    precision = metrics.precision_score(ytest, ypred, average='micro')
    print("Precision pour l'algo %s : %f"%(algo,precision))