from sklearn.datasets import fetch_openml
from sklearn import neighbors
from sklearn import model_selection

import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784')

# -------------------------------------------
# Comprendre des propriétés du dataset mnist
# -------------------------------------------
print(mnist)
print(mnist.data)
print(mnist.target)
len(mnist.data)
help(len)
print(mnist.data.shape)
print(mnist.target.shape)
mnist.data[0]
mnist.data[0][1]
mnist.data[:,1]
mnist.data[:100]


# -------------------------------------------
# Visualiser les données
# -------------------------------------------
# Affiche l'image
images = mnist.data.reshape((-1, 28, 28))
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()

# Affiche la classe de l'image
print("Classe de l'image : " + str(mnist.target[0]))


# -------------------------------------------
# Explorer d'autres jeux de données
# -------------------------------------------
mauna = fetch_openml('mauna-loa-atmospheric-co2')
print(mauna)
print(mauna.data)
print(mauna.target)
len(mauna.data)
help(len)
print(mauna.data.shape)
print(mauna.target.shape)
mauna.data[0]
mauna.data[0][1]
mauna.data[:,1]
mauna.data[:100]