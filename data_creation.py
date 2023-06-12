import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# https://habr.com/ru/companies/sberbank/articles/716172/
# Cгенерируем синтетические данные. Создадим два набора данных. 
# В одном из них целевая переменная находится в линейной зависимости с независимыми факторами, 
# а в другом зависимость между целевой и факторами сильно нелинейная.

print('Генерируем синтетические данные для линейной зависимости...')

X, y = make_regression(
    # количество объектов
    n_samples=500,
    # количество признаков
    n_features=10,
    # количество информативных признаков
    n_informative=5,
    random_state=0,
    # перемешивание признаков shuffle=True, иначе значащие стоят впереди
    shuffle=False
)

#  Разобьём выборку на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.33, 
    random_state=42
)

# Сохраняем данные в файлы
np.savetxt("train/linear_X_train.csv", X_train, fmt="%0.2f", delimiter=",")
np.savetxt("train/linear_y_train.csv", y_train, fmt="%0.2f", delimiter=",")
np.savetxt("test/linear_X_test.csv", X_test, fmt="%0.2f", delimiter=",")
np.savetxt("test/linear_y_test.csv", y_test, fmt="%0.2f", delimiter=",")
print('Данные записаны в файлы: train/linear_X_train.csv ; train/linear_X_train.csv; test/linear_X_test.csv; test/linear_y_test.csv')


"""
Генерируем кластеры нормально распределённых точек в вершинах гиперкуба. 
Количество вершин куба совпадает с количеством информативных признаков. 
Это вводит взаимозависимость между этими признаками и добавляет к данным различные виды дополнительного шума.
"""
print('Генерируем синтетические данные: кластеры точек в вершинах гиперкуба...')
X, y = make_classification(
    n_samples=500, 
    n_features=20, 
    n_informative=5, 
    n_classes=2,
    # 2 * class_sep = длина стороны гиперкуба
    class_sep=1.0, 
    hypercube=True,
    shuffle=False,
    random_state=13
)

#  Разобьём выборку на обучающую и тестовую
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y, 
    test_size=0.33, 
    random_state=42
)

# Сохраняем данные в файлы
np.savetxt("train/hypercube_X_train.csv", X_train, fmt="%0.2f", delimiter=",")
np.savetxt("train/hypercube_y_train.csv", y_train, fmt="%0.2f", delimiter=",")
np.savetxt("test/hypercube_X_test.csv", X_test, fmt="%0.2f", delimiter=",")
np.savetxt("test/hypercube_y_test.csv", y_test, fmt="%0.2f", delimiter=",")
print('Данные записаны в файлы: train/hypercube_X_train.csv ; train/hypercube_X_train.csv; test/hypercube_X_test.csv; test/hypercube_y_test.csv')




