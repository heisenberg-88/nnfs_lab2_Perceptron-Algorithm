from sklearn import datasets
import numpy as np
from Perceptron import perceptron

x = np.genfromtxt('nonlinsep.csv',delimiter=',',usecols=(1,2))
y = np.genfromtxt('nonlinsep.csv',delimiter=',',usecols=(0))

theta , missed_list = perceptron(x,y,0.5,100)
print(missed_list)

# x, y = datasets.make_blobs(n_samples=1500,n_features=2,
#                            centers=2,cluster_std=1.06,
#                            random_state=2)


# x, y = datasets.make_blobs(n_samples=1500,n_features=2,
#                            centers=2,cluster_std=1.3,
#                            random_state=2)

# x = np.genfromtxt('data.csv',delimiter=',',usecols=(1,2))
# y = np.genfromtxt('data.csv',delimiter=',',usecols=(0))







