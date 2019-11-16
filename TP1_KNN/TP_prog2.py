import numpy as np
from sklearn.datasets import fetch_openml
from sklearn import neighbors
from sklearn import model_selection
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784')

value_nb = 5000
vect = np.random.randint(70000, size=value_nb)
data = mnist.data[vect]
target = mnist.target[vect]

k = 10
percent = 0.8
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
clf = neighbors.KNeighborsClassifier(k)
clf.fit(xtrain, ytrain)

print("classe de l'image 4 =")
print(ytest[4])
print("classe prédite de l'image 4 =")
print(clf.predict(xtest)[4])

print(clf.score(xtest, ytest))

# méthode : k-fold cross validation
def get_score_with_kfold(dataset, n_fold, n_neighbors):
    kf = model_selection.KFold(n_splits=n_fold, shuffle=True)
    num_fold = 0
    score = 0
    for train_index, test_index in kf.split(dataset):
        num_fold += 1
        xtrain = data[train_index]
        ytrain = target[train_index]
        xtest = data[test_index]
        ytest = target[test_index]
        clf = neighbors.KNeighborsClassifier(n_neighbors)
        clf.fit(xtrain, ytrain)
        score += clf.score(xtest, ytest)
    score /= num_fold
    return score

# faire varier le nombre de voisins k
"""scores = []
for k in range (2,16):
    score = get_score_with_kfold(data, 10, k)
    print('Score pour %d voisins : %f'%(k, score))
    scores.append(score)

plt.plot(range(2,16), scores)
plt.xlabel("Nombre de voisins k")
plt.ylabel("Score")
plt.title("Estimation de la fiabilité en fonction de k :")
plt.show()"""

# faire varier le pourcentage
k = 6
scores = []
for percent in np.arange(0.1, 1, 0.1):
    xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=percent)
    clf = neighbors.KNeighborsClassifier(k)
    clf.fit(xtrain, ytrain)
    score = clf.score(xtest, ytest)
    print('Echantillon training : %f, score: %f'%(percent, score))
    scores.append(score)

plt.plot(np.arange(0.1, 1, 0.1), scores)
plt.xlabel("Pourcentage de l'échantillon training")
plt.ylabel("Score")
plt.title("Fiabilité en fonction du pourcentage de l'échantillon de training :")
plt.show()
