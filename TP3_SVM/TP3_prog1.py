from sklearn import metrics
from sklearn import model_selection
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC

import matplotlib.pyplot as plt
import numpy as np
import time


mnist = fetch_openml('mnist_784')

# --------------------------------------------------------------------
# Test avec un training à 70% pour l’apprentissage et 30% pour les tests
# --------------------------------------------------------------------

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(minst.data, minst.target, train_size=0.7)

# ----------------- Variation de la fonction noyau -----------------
#TODO check for precomputed vers
for noyau in ['linear','poly','rbf','sigmoid','precomputed']:
	classifier = SVC(gamma='auto', kernel=noyau) 
	classifier.fit(xtrain, ytrain)
	predicted = classifier.predict(xtest)
	score = classifier.score(xtest, ytest)
	print("Score : " + str(score))

# ------ Variation du paramètre de tolerance aux erreurs C ---------
erreur = []
for tolerancce in [0.1,0.3,0.5,0.7,1.0]:
    clsvm = SVC(gamma='auto', kernel='poly' ,C=tolerancce)
    clsvm.fit(xtrain, ytrain)
    predicted = clsvm.predict(xtest) 
    erreur.append(metrics.zero_one_loss(ytest, predicted))
print(erreur)
plt.plot([0.1,0.3,0.5,0.7,1.0], erreur)
plt.xlabel("Valeur de C")
plt.ylabel("Erreur de classification")
plt.title("Erreur de classification sur les données d’entrainement et de test en fonction de C")
plt.show()


#TODO
cm = confusion_matrix(ytest, Y_pred)
print(cm)

# -------------------------------- Variation de la fonction noyau et de C --------------------------------
# Analyse du temps d'apprentissage, de la précision, du rappel, de l'erreur et de la matrice de confusion 

# il faut prendre le temps en cpu et non en temps "normal" (ne pas tenir compte de cette remarque pour TP1): time.clock()
for noyau in ['linear','poly','rbf','sigmoid','precomputed']:
    erreur = []
    temps = []
    precision = []
    rappel = []
    for tolerancce in [0.1,0.3,0.5,0.7,1.0]:
        time_start = time.process_time
        clsvm = SVC(kernel=noyau, C=tolerancce)
        clsvm.fit(xtrain, ytrain)
        predicted = clsvm.predict(xtest) 
        time_stop = time.process_time
        temps.append(time_stop-time_start)
        erreur.append(metrics.zero_one_loss(ytest, predicted))
        precision.append(metrics.precision_score(ytest, predicted,average='micro'))
        rappel.append(metrics.recall_score(ytest, predicted,average='micro'))
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
			print("Noyau : "+ noyau + " / "+ type_analyse +" : " + str(content))
			plt.plot([0.1,0.3,0.5,0.7,1.0], erreur)
			plt.xlabel("Valeur de C")
			plt.ylabel(ylabel)
			plt.title("Noyau : "+ noyau + " / "+ type_analyse +" en fonction de C")
			plt.show()



#accès cluster : ssh nom@srv-ens-calcul.insa-touluse.fr
#Todo : matrice de confusion pour toutes les méthodes  + precomputed pour SVM



