from sklearn.datasets import fetch_openml
from sklearn import model_selection
from sklearn import neural_network
from sklearn import metrics

import math
import numpy as np
import time

mnist = fetch_openml('mnist_784')
data = mnist.data
target = mnist.target
percent = 0.7

# --------------------------------------------------------------------
# Test avec un training à 70% pour l’apprentissage et 30% pour les tests
# --------------------------------------------------------------------

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)


clf = neural_network.MLPClassifier(hidden_layer_sizes=(50,))
clf.fit(xtrain, ytrain)
ypred = clf.predict(xtest)

precision = metrics.precision_score(ytest, ypred, average='micro')
print("Precision :" + str(precision))

print("Classe de l'image 4 : " + str(ytest[4]))
print("Classe prédite de l'image 4 : " + str(clf.predict(xtest)[4]))


# ----------------- Variation du nombre de couches  -----------------
tup = ()
for nb_couches in range(1,101,10):
    count = 0
    while(count < nb_couches): 
        tup += (50,)
        count += 1
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup)
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)

    precision = metrics.precision_score(ytest, ypred, average='micro')
    print("Precision pour {} couches de 50 neurones : {}".format(nb_couches,precision))


# ---------------- Variation du nombre de neurones  ----------------
nb_couches = [2, 10, 20, 50, 100]
for nb in nb_couches :
    tup = ()
    for nb_neurones in np.linspace(60, 11, nb, endpoint=False):
        tup += (math.ceil(nb_neurones),)
    print("tup = {}".format(tup))
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup)
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    precision = metrics.precision_score(ytest, ypred, average='micro')
    print("Precision pour {} couches de 60 à 11 neurones : {}".format(nb, precision))



# ------ Etude la convergence des algorithmes d’optimisation   --------
tup = ()
count = 0
while(count < 10): 
    tup += (50,)
    count +=1
print(tup)

for algo in ['lbfgs', 'sgd', 'adam']:
    
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup, solver=algo)
    
    time_start = time.process_time() # On regarde le temps CPU
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    time_stop = time.process_time()
    temps = time_stop-time_start
    precision = metrics.precision_score(ytest, ypred, average='micro')
    erreur = metrics.zero_one_loss(ytest, ypred)
    precision = metrics.precision_score(ytest, ypred,average='micro')
    rappel = metrics.recall_score(ytest, ypred,average='micro')
    print("Algo : "+ str(algo))
    for type_analyse in ["Erreur", "Temps", "Précision", "Rappel"]:
        if(type_analyse == "Erreur"):
            content = erreur
        elif(type_analyse == "Temps"):
            content = temps
        elif(type_analyse == "Precision"):
            content = precision
        else:
            content = rappel
        print(type_analyse +" : " + str(content))


# ------------- Variation des fonctions d’activation   ----------------
tup = ()
count = 0
while(count < 10): 
    tup += (50,)
    count +=1
print(tup)

for function in ['identity', 'logistic', 'tanh', 'relu']:
    
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup, activation=function)
    time_start = time.process_time() # On regarde le temps CPU
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    time_stop = time.process_time()
    temps = time_stop-time_start
    precision = metrics.precision_score(ytest, ypred, average='micro')
    erreur = metrics.zero_one_loss(ytest, ypred)
    precision = metrics.precision_score(ytest, ypred,average='micro')
    rappel = metrics.recall_score(ytest, ypred,average='micro')
    print("Fonction d'activation : "+ str(function))
    for type_analyse in ["Erreur", "Temps", "Précision", "Rappel"]:
        if(type_analyse == "Erreur"):
            content = erreur
        elif(type_analyse == "Temps"):
            content = temps
        elif(type_analyse == "Precision"):
            content = precision
        else:
            content = rappel
        print(type_analyse +" : " + str(content))


# ------- Variation de la valeur de la régularisation L2 --------------
tup = ()
count = 0
while(count < 10): 
    tup += (50,)
    count +=1
print(tup)

erreur = []
temps = []
precision = []
rappel = []

for a in np.linspace(0.0001, 0.0005, 5, endpoint=True):
    
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup, alpha=a)
    time_start = time.process_time() # On regarde le temps CPU
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    time_stop = time.process_time()
    temps.append(time_stop-time_start)
    erreur.append(metrics.zero_one_loss(ytest, ypred))
    precision.append(metrics.precision_score(ytest, ypred,average='micro'))
    rappel.append(metrics.recall_score(ytest, ypred,average='micro'))

for type_analyse in ["Erreur", "Temps", "Précision", "Rappel"]:
    if(type_analyse == "Erreur"):
        content = erreur
        ylabel = "Erreur de classification"
    elif(type_analyse == "Temps"):
        content = temps
        ylabel = "Temps d'apprentissage"
    elif(type_analyse == "Precision"):
        content = precision
        ylabel = "Precision"
    else:
        content = rappel
        ylabel = "Rappel"
    print(type_analyse +" : " + str(content))
    plt.plot(np.linspace(0.0001, 1, 20, endpoint=True), content)
    plt.xlabel("Valeur de alpha")
    plt.ylabel(ylabel)
    plt.title(type_analyse +" en fonction de alpha")
    plt.show()
