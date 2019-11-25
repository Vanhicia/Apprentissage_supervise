from sklearn import metrics
from sklearn import model_selection
from sklearn import neighbors
from sklearn import neural_network
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import math
import matplotlib.pyplot as plt
import numpy as np
import time


mnist = fetch_openml('mnist_784')

# --------------------------------------------------------------------
# Test avec un training à 80% pour l’apprentissage et 30% pour les tests
# --------------------------------------------------------------------

percent = 0.8
variation = range(5000, 10000, 500)


# ----------------- Méthode pour afficher les courbes -----------------
def afficher_courbe(erreur,temps,precision,rappel,cm):
    for type_analyse in ["Erreur", "Temps", "Précision", "Rappel", "Matrice de confusion"]:
        if(type_analyse == "Erreur"):
            content = erreur
            ylabel = "Erreur de classification"
        elif(type_analyse == "Temps"):
            content = temps
            ylabel = "Temps d'apprentissage"
        elif(type_analyse == "Précision"):
            content = precision
            ylabel = "Précision"
        elif(type_analyse == "Rappel"):
            content = rappel
            ylabel = "Rappel"
        else:
            print("Matrice de confusion")
            print(cm)

        if(type_analyse != "Matrice de confusion"):
            print(type_analyse +" : " + str(content))
            plt.plot(variation, content)
            plt.xlabel("Taille de l'échantillon")
            plt.ylabel(ylabel)
            plt.title(type_analyse +" en fonction de la taille de l'échantillon")
            plt.show()
            
            
# ----------------- Test avec la méthode KNN -----------------
erreur = []
temps = []
precision = []
rappel = []
cm = []
for taille in variation:
    index_vect = np.random.randint(70000, size=taille)
    data = mnist.data[index_vect]
    target = mnist.target[index_vect]
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clf = neighbors.KNeighborsClassifier(n_neighbors=3, p=4, n_jobs=-1)
    time_start = time.process_time() # On regarde le temps CPU
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    time_stop = time.process_time()
    temps.append(time_stop-time_start)
    erreur.append(metrics.zero_one_loss(ytest, ypred))
    precision.append(metrics.precision_score(ytest, ypred,average='micro'))
    rappel.append(metrics.recall_score(ytest, ypred,average='micro'))
    cm.append(confusion_matrix(ytest, ypred))
afficher_courbe(erreur,temps,precision,rappel,cm)

    
# ----------------- Test avec la méthode MLP -----------------
erreur = []
temps = []
precision = []
rappel = []
cm = []

nb_couches = 10
tup = ()
count = 0
for nb_neurones in np.linspace(60, 11, nb_couches, endpoint=True):
    tup += (math.ceil(nb_neurones),)
print("tup = {}".format(tup))
    
for taille in variation:
    index_vect = np.random.randint(70000, size=taille)
    data = mnist.data[index_vect]
    target = mnist.target[index_vect]
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clf = neural_network.MLPClassifier(hidden_layer_sizes=tup, solver='adam', activation='relu',alpha=0.0001)
    time_start = time.process_time() # On regarde le temps CPU
    clf.fit(xtrain, ytrain)
    ypred = clf.predict(xtest)
    time_stop = time.process_time()
    temps.append(time_stop-time_start)
    erreur.append(metrics.zero_one_loss(ytest, ypred))
    precision.append(metrics.precision_score(ytest, ypred,average='micro'))
    rappel.append(metrics.recall_score(ytest, ypred,average='micro'))
    cm.append(confusion_matrix(ytest, ypred))
afficher_courbe(erreur,temps,precision,rappel,cm)
        

# ----------------- Test avec la méthode SVM -----------------
erreur = []
temps = []
precision = []
rappel = []
cm = []
for taille in variation:
    index_vect = np.random.randint(70000, size=taille)
    data = mnist.data[index_vect]
    target = mnist.target[index_vect]
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clsvm = SVC(gamma='auto', kernel='poly', C=0.1)
    time_start = time.process_time() # On regarde le temps CPU
    clsvm.fit(xtrain, ytrain)
    ypred = clsvm.predict(xtest) 
    time_stop = time.process_time()
    temps.append(time_stop-time_start)
    erreur.append(metrics.zero_one_loss(ytest, ypred))
    precision.append(metrics.precision_score(ytest, ypred,average='micro'))
    rappel.append(metrics.recall_score(ytest, ypred,average='micro'))
    cm.append(confusion_matrix(ytest, ypred))
afficher_courbe(erreur,temps,precision,rappel,cm)





