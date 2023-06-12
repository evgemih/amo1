import numpy as np
from sklearn.preprocessing import StandardScaler

X_train = np.loadtxt("train/linear_X_train.csv", delimiter=',')
#y_train = np.loadtxt("train/linear_y_train.csv", delimiter=',')
X_test = np.loadtxt("test/linear_X_test.csv", delimiter=',')
#y_test = np.loadtxt("test/linear_y_test.csv", delimiter=',')

print('Cтандартизуем данные...')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = np.loadtxt("test/linear_X_test.csv", delimiter=',')

print('Сохраняем данные в файлы train/linear_X_train_std.csv; test/linear_X_test_std.csv') 
np.savetxt("train/linear_X_train_std.csv", X_train, fmt="%0.2f", delimiter=",")
np.savetxt("test/linear_X_test_std.csv", X_test, fmt="%0.2f", delimiter=",")

X_train = np.loadtxt("train/hypercube_X_train.csv", delimiter=',')
#y_train = np.loadtxt("train/hypercube_y_train.csv", delimiter=',')
X_test = np.loadtxt("test/hypercube_X_test.csv", delimiter=',')
#y_test = np.loadtxt("test/hypercube_y_test.csv", delimiter=',')

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = np.loadtxt("test/hypercube_X_test.csv", delimiter=',')

print('Сохраняем данные в файлы train/hypercube_X_train_std.csv; test/hypercube_X_test_std.csv') 
np.savetxt("train/hypercube_X_train_std.csv", X_train, fmt="%0.2f", delimiter=",")
np.savetxt("test/hypercube_X_test_std.csv", X_test, fmt="%0.2f", delimiter=",")

