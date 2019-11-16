from sklearn.datasets import fetch_openml
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import model_selection

mnist = fetch_openml('mnist_784')

#print(mnist)
#print (mnist.data)
#print (mnist.target)
#print(len(mnist.data))
#help(len)
#print (mnist.data.shape)
#print (mnist.target.shape)
#print(mnist.data[0])
#print(mnist.data[0][1])
#print(mnist.data[:,1])
#print(mnist.data[:100])

images = mnist.data.reshape((-1, 28, 28))
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest")
#plt.show()
#affiche la classe de l'image
#print(mnist.target[0])

k = 5
percent = 0.7
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(mnist.data, mnist.target, train_size=percent)
clf = neighbors.KNeighborsClassifier (k)
clf.fit(xtrain, ytrain)
print("train OK")
clf.predict(xtest)
clf.predict_proba(xtest)
clf.score(xtest, ytest)