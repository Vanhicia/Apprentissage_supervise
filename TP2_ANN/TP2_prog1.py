from sklearn.datasets import fetch_openml
from sklearn import model_selection
import numpy as np
from sklearn.neural_network imoort MLPClassifier

mnist = fetch_openml('mnist_784')

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(data, target, train_size=0.8)

# varier 10 fois le nombre de couches
# 7 neurones par couches
# puis diminuer le nombre de neurones sur les dernières couches : 50 sur la première couvhe et 10 à la fin

clf = MLPClassifier()hidden_layer_sizes = (50))
