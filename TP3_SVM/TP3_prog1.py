from sklearn import metrics
from sklearn import model_selection
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC

import numpy as np
import time


mnist = fetch_openml('mnist_784')


percent = 0.7
index_vect = np.random.randint(70000, size=5000)
data = mnist.data[index_vect]
target = mnist.target[index_vect]

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)


classifier = SVC(gamma='auto') 
classifier.fit(xtrain, ytrain)
predicted = classifier.predict(xtest)
score = classifier.score(xtest, ytest)
print("Score : " + str(score))



clsvm = SVC(kernel='linear')
clsvm.fit(xtrain, ytrain)
predicted = clsvm.predict(xtest)
score = clsvm.score(xtest, ytest)
print("Score : " + str(score))


clsvm = SVC(kernel='poly')
clsvm.fit(xtrain, ytrain)
predicted = clsvm.predict(xtest)
score = clsvm.score(xtest, ytest)
print("Score : " + str(score))

clsvm = SVC(kernel='rbf')
clsvm.fit(xtrain, ytrain)
predicted = clsvm.predict(xtest)
score = clsvm.score(xtest, ytest)
print("Score : " + str(score))


clsvm = SVC(kernel='sigmoid')
clsvm.fit(xtrain, ytrain)
predicted = clsvm.predict(xtest)
score = clsvm.score(xtest, ytest)
print("Score : " + str(score))


#TODO
clsvm = SVC(kernel='precomputed')
clsvm.fit(xtrain, ytrain)
predicted = clsvm.predict(xtest)
score = clsvm.score(xtest, ytest)
print("Score : " + str(score))


erreur = []
for tolerancce in [0.1,0.3,0.5,0.7,1.0]:
    clsvm = SVC(gamma='auto', kernel='poly' ,C=tolerancce)
    clsvm.fit(xtrain, ytrain)
    predicted = clsvm.predict(xtest) 
    erreur.append(metrics.zero_one_loss(ytest, predicted))

print(erreur)
import matplotlib.pyplot as plt
plt.plot([0.1,0.3,0.5,0.7,1.0], erreur)
plt.xlabel("Valeur de C")
plt.ylabel("Erreur de classification")
plt.title("Erreur de classification sur les données d’entrainement et de test en fonction de C")
plt.show()


#TODO
cm = confusion_matrix(ytest, Y_pred)
print(cm)



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
    print("Noyau : "+ noyau + " / Erreur : " + str(erreur))
    plt.plot([0.1,0.3,0.5,0.7,1.0], erreur)
    plt.xlabel("Valeur de C")
    plt.ylabel("Erreur de classification")
    plt.title("Noyau : "+ noyau + " / Erreur en fonction de C")
    plt.show()
    
    print("Noyau : "+ noyau + " / Temps : " + str(temps))
    plt.plot([0.1,0.3,0.5,0.7,1.0], temps)
    plt.xlabel("Valeur de C")
    plt.ylabel("Temps d'apprentissage")
    plt.title("Noyau : "+ noyau + " / Temps en fonction de C")
    plt.show()
    
    print("Noyau : "+ noyau + " / Précision : " + str(precision))
    plt.plot([0.1,0.3,0.5,0.7,1.0], precision)
    plt.xlabel("Valeur de C")
    plt.ylabel("Precision")
    plt.title("Noyau : "+ noyau + " / Precision en fonction de C")
    plt.show()
    
    print("Noyau : "+ noyau + " / Rappel : " + str(rappel))
    plt.plot([0.1,0.3,0.5,0.7,1.0], rappel)
    plt.xlabel("Valeur de C")
    plt.ylabel("Rappel")
    plt.title("Noyau : "+ noyau + " / Rappel en fonction de C")
    plt.show()



#accès cluster : ssh nom@srv-ens-calcul.insa-touluse.fr
#Todo : matrice de confusion pour toutes les méthodes  + precomputed pour SVM



