
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

print('Загружаем линейную модель из файла: model/linear_model.pkl')
with open('model/linear_model.pkl', 'rb') as f:
    clf = pickle.load(f)

X_test = np.loadtxt("test/linear_X_test_std.csv", delimiter=',')
y_test = np.loadtxt("test/linear_y_test.csv", delimiter=',')
    
print('Точность модели на тестовых данных, score = ', clf.score(X_test, y_test))


print('\nЗагружаем нелинейную модель из файла: model/RFC_model.pkl')
with open('model/RFC_model.pkl', 'rb') as f:
    clf = pickle.load(f)

X_test = np.loadtxt("test/hypercube_X_test_std.csv", delimiter=',')
y_test = np.loadtxt("test/hypercube_y_test.csv", delimiter=',')

print('Точность модели на тестовых данных, accuracy_score = ',accuracy_score(y_test, clf.predict(X_test)))


