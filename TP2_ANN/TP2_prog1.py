from sklearn.datasets import fetch_openml
from sklearn import model_selection
from sklearn import neural_network
from sklearn import metrics

import numpy as np


mnist = fetch_openml('mnist_784')


# --------------------------------------------------------------------
# Test avec un training à 70% pour l’apprentissage et 30% pour les tests
# --------------------------------------------------------------------
percent = 0.7
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(mnist.data, mnist.target, train_size=percent)


clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,))
clf.fit(xtrain, ytrain)
ypred = clf.predict(xtest)

precision = metrics.precision_score(ytest, ypred, average='micro')
print("Precision :" + str(precision))

print("Classe de l'image 4 : " + str(ytest[4]))
print("Classe prédite de l'image 4 : " + str(clf.predict(xtest)[4]))


# ----------------- Variation du nombre de couches  -----------------
tup = ()
for k in range(1,101,10):
    tup += (50,) # TO DO : corriger tup !!!!
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)

    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup)
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)

    precision = metrics.precision_score(ytest, ypred, average='micro')
    print("Precision pour %d couches de 50 neurones : %f" %(k,precision))

		
# ---------------- Variation du nombre de neurones  ----------------
nb_couches = [2, 10, 20, 50, 100]
tup = ()
for nb in nb_couches :
	for nb_neurones in np.linspace(60, 11, nb, endpoint=False):
    tup += (nb_neurones,)
	print("tup = {}".format(tup))
	xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
	clf = neural_network.MLPClassifier(hidden_layer_sizes=tup)
	clf.fit(xtrain, ytrain)
	ypred = clf.predict(xtest)
	precision = metrics.precision_score(ytest, ypred, average='micro')
	print("Precision pour {} couches de 60 à 11 neurones : {}".format(nb, precision))


# ------ Etude la convergence des algorithmes d’optimisation   --------
for algo in ['lbfgs', 'sgd', 'adam']:
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup, solver=algo)
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    precision = metrics.precision_score(ytest, ypred, average='micro')
    print("Precision pour l'algo {} : {}".format(algo,precision))


# ------------- Variation des fonctions d’activation   ----------------
for function in [‘identity’, ‘logistic’, ‘tanh’, ‘relu’]:
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup, activation=function)
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    precision = metrics.precision_score(ytest, ypred, average='micro')
    print("Precision pour l'algo {} : {}".format(algo,precision))


# ------- Variation de la valeur de la régularisation L2 --------------



# Analyse du temps d'apprentissage, de la précision, du rappel et de l'erreur