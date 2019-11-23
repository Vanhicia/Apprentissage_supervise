from sklearn import metrics
from sklearn import model_selection
from sklearn.datasets import fetch_openml
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np
import time


mnist = fetch_openml('mnist_784')
index_vect = np.random.randint(70000, size=5000)
data = mnist.data[index_vect]
target = mnist.target[index_vect]

# --------------------------------------------------------------------
# Test avec un training à 70% pour l’apprentissage et 30% pour les tests
# --------------------------------------------------------------------

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.7)

# ----------------- Variation de la fonction noyau -----------------
for noyau in ['linear','poly','rbf','sigmoid']:
    classifier = SVC(gamma='auto', kernel=noyau)
    classifier.fit(xtrain, ytrain)
    ypredict = classifier.predict(xtest)
    score = classifier.score(xtest, ytest)
    print("Noyau : "+ noyau +" / Score : " + str(score))
		
		
# ------ Variation du paramètre de tolerance aux erreurs C ---------
erreur = []
cm = []
for tolerancce in [0.1,0.3,0.5,0.7,1.0]:
    clsvm = SVC(gamma='auto', kernel='poly' ,C=tolerancce)
    clsvm.fit(xtrain, ytrain)
    ypredict = clsvm.predict(xtest) 
    erreur.append(metrics.zero_one_loss(ytest, ypredict))
    cm.append(confusion_matrix(ytest, ypredict))
print(erreur)
plt.plot([0.1,0.3,0.5,0.7,1.0], erreur)
plt.xlabel("Valeur de C")
plt.ylabel("Erreur de classification")
plt.title("Erreur de classification sur les données d’entrainement et de test en fonction de C")
plt.show()

print("Matrice de confusion")
print(cm)



# -------------------------------- Variation de la fonction noyau et de C --------------------------------
# Analyse du temps d'apprentissage, de la précision, du rappel, de l'erreur et de la matrice de confusion 


# Variation de la fonction noyau
for noyau in ['linear','poly','rbf','sigmoid']:
    erreur = []
    temps = []
    precision = []
    rappel = []
    cm = []

    # Variation de C
    for tolerancce in [0.1,0.3,0.5,0.7,1.0]:
        time_start = time.process_time() # On regarde le temps CPU
        clsvm = SVC(gamma='auto', kernel=noyau, C=tolerancce)
        clsvm.fit(xtrain, ytrain)
        ypredict = clsvm.predict(xtest) 
        time_stop = time.process_time()
        temps.append(time_stop-time_start)
        erreur.append(metrics.zero_one_loss(ytest, ypredict))
        precision.append(metrics.precision_score(ytest, ypredict,average='micro'))
        rappel.append(metrics.recall_score(ytest, ypredict,average='micro'))
        cm.append(confusion_matrix(ytest, ypredict))

    # Affichage des différentes courbes
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
            print("Noyau : "+ noyau + " / "+ type_analyse +" : " + str(content))
            plt.plot([0.1,0.3,0.5,0.7,1.0], content)
            plt.xlabel("Valeur de C")
            plt.ylabel(ylabel)
            plt.title("Noyau : "+ noyau + " / "+ type_analyse +" en fonction de C")
            plt.show()