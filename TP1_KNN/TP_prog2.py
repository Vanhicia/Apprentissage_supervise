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

# faire varier le nombre de voisins k
# méthode : k-fold cross validation
kf = model_selection.KFold(n_splits=10, shuffle=True)
scores = []
for k in range (2,16):
    print('k: ', k)
    num_fold = 0
    score = 0
    for train_index, test_index in kf.split(data):
        num_fold += 1
        print('Start KFold number %d from %d'%(num_fold, 10))
        print('Split train: ', len(train_index))
        print('Split valid: ', len(test_index))
        xtrain = data[train_index]
        ytrain = target[train_index]
        xtest = data[test_index]
        ytest = target[test_index]
        clf = neighbors.KNeighborsClassifier(k)
        clf.fit(xtrain, ytrain)
        score += clf.score(xtest, ytest)
        print('score =', clf.score(xtest, ytest))
    score /= num_fold
    scores.append(score)

plt.plot(range(2,16),scores)
plt.xlabel("Nombre de voisins k")
plt.ylabel("Score")
plt.title("Estimation de la fiabilité en fonction de k :")

