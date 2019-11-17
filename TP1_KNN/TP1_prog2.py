from sklearn.datasets import fetch_openml
from sklearn import neighbors
from sklearn import model_selection
from time import time

import matplotlib.pyplot as plt
import numpy as np

mnist = fetch_openml('mnist_784')
index_vect = np.random.randint(70000, size=5000)
data = mnist.data[index_vect]
target = mnist.target[index_vect]


# --------------------------------------------------------------------
# Test avec une BDD à 80% pour l’apprentissage et 20% pour les tests
# --------------------------------------------------------------------

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.8)
'''
# ----------------- Test avec K = 10 -----------------
clf = neighbors.KNeighborsClassifier(10)
clf.fit(xtrain, ytrain)

print("Classe de l'image 4 : " + str(ytest[4]))
print("Classe prédite de l'image 4 : " + str(clf.predict(xtest)[4]))

score = clf.score(xtest, ytest)
print("Score : " + str(score))
print("Taux d'erreur : " + str((1-score)*100) + " %")

# --------- Méthode : k-fold cross validation -------- 
def get_score_with_kfold(data, n_fold, k):
	kf = model_selection.KFold(n_splits=n_fold, shuffle=True)
	num_fold = 0
	score = 0
	for train_index, test_index in kf.split(data):
		num_fold += 1
		xtrain = data[train_index]
		ytrain = target[train_index]
		xtest = data[test_index]
		ytest = target[test_index]
		clf = neighbors.KNeighborsClassifier(k)
		clf.fit(xtrain, ytrain)
		score += clf.score(xtest, ytest)
	score /= num_fold
	return score



# ---- Variation du nombre de voisins k de 2 à 15 ----
scores = []
for k in range(2,16):
	score = get_score_with_kfold(data, 10, k)
	print("Score pour {} voisins : {}".format(k, score))
	scores.append(score)

plt.plot(range(2,16), scores)
plt.xlabel("Nombre de voisins k")
plt.ylabel("Score")
plt.title("Estimation de la fiabilité en fonction de k :")
plt.show()


# ------ Variation du pourcentage training/test -----
k = 6
scores = []
for percent in np.arange(0.1, 1, 0.1):
	xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
	clf = neighbors.KNeighborsClassifier(k)
	clf.fit(xtrain, ytrain)
	score = clf.score(xtest, ytest)
	print("Echantillon training : {} / Score : {}".format(percent, score))
	scores.append(score)
    

plt.plot(np.arange(0.1, 1, 0.1), scores)
plt.xlabel("Pourcentage de l'échantillon training")
plt.ylabel("Score")
plt.title("Fiabilité en fonction du pourcentage de l'échantillon de training :")
plt.show()


# ---- Variation de la taille de l'échantillon training ----
k = 6
scores = []
tailles = []
for taille in range(5000, 15000, 1000):
	index_vect = np.random.randint(70000, size=taille)
	data = mnist.data[index_vect]
	target = mnist.target[index_vect]
	xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.8)
	clf = neighbors.KNeighborsClassifier(k)
	clf.fit(xtrain, ytrain)
	score = clf.score(xtest, ytest)
	print("Taille de l'échantillon training : {} / Score : {}".format(taille, score))
	scores.append(score)
	tailles.append(taille)

plt.plot(tailles, scores)
plt.xlabel("Taille de l'échantillon training")
plt.ylabel("Score")
plt.title("Fiabilité en fonction de la taille de l'échantillon training :")
plt.show()


# -------- Variation des types de distance p -------
k = 6
scores = []
index_vect = np.random.randint(70000, size=5000)
data = mnist.data[index_vect]
target = mnist.target[index_vect]
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.8)

for p in range(1, 11):
	clf = neighbors.KNeighborsClassifier(k, p=p)
	clf.fit(xtrain, ytrain)
	score = clf.score(xtest, ytest)
	print("Type de distance p : {} / Score : {}".format(p, score))
	scores.append(score)

plt.plot(range(1,11), scores)
plt.xlabel("Type de distance p")
plt.ylabel("Score")
plt.title("Fiabilité en fonction du type de distance :")
plt.show()

'''
# -------- Analyse du temps pour n_job à 1 et -1 -------
for i in [-1,1]:
	time_start = time()
	clf = neighbors.KNeighborsClassifier(3,n_jobs=i)
	clf.fit(xtrain, ytrain)
	score = clf.score(xtest, ytest)
	time_stop = time()
	print("Valeur de n_jobs : {} / Temps total : {}".format(i,time_stop-time_start))