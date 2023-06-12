'''
Источник: https://habr.com/ru/companies/sberbank/articles/716172/

Можно отобрать значащие признаки при помощи регрессии по методу Лассо. 
Данная модель использует L1-регуляризацию для назначения штрафов незначащим весам. 
В результате обучения коэффициенты незначащих признаков становятся равны нулю, 
что позволяет проводить отбор признаков. 
В этом отличие от L2-регуляризации, в которой коэффициенты признаков могут иметь значение, 
близкое к нулю, но не равное ему. 
Гиперпараметром для модели с данной регуляризацией выступает вещественный коэффициент α. 
Чем больше данный коэффициент, тем больше признаков будет отсеяно.

Помимо отбора признаков данная модель позволяет сравнивать значимость признаков между собой. 
Но для этого данные необходимо стандартизировать, так как изначально они могут быть измерены в разных шкалах.

'''

import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

X_train = np.loadtxt("train/linear_X_train_std.csv", delimiter=',')
y_train = np.loadtxt("train/linear_y_train.csv", delimiter=',')
X_test = np.loadtxt("test/linear_X_test_std.csv", delimiter=',')
y_test = np.loadtxt("test/linear_y_test.csv", delimiter=',')

print('Обучение для линейной зависимости...')

# Создаем объект линейной регрессии
clf = linear_model.Lasso(alpha=0.1)

# Обучаем модель на созданных данных
clf.fit(X_train, y_train)

# Проверяем, сколько и какие признаки модель посчитала информативными:
count = 0
for i in clf.coef_:
    if i != 0:
        count += 1

print(f'Модель отобрала {count} признаков; настоящее количество информативных признаков: 5')

# Посмотрим, те ли коэффициенты модель посчитала информативными
print('Значения коэффициентов признаков: ',clf.coef_)

# print('score = ', clf.score(X_test, y_test))

print('Сохраняем модель в файл: model/linear_model.pkl')
with open('model/linear_model.pkl','wb') as f:
    pickle.dump(clf,f)

'''
=========================================================================================================

Для нелинейных зависимостей (сгенерированных кластеры нормально распределённых точек в вершинах гиперкуба)
хорошо справляется алгоритм случайного леса
'''

X_train = np.loadtxt("train/hypercube_X_train_std.csv", delimiter=',')
y_train = np.loadtxt("train/hypercube_y_train.csv", delimiter=',')
X_test = np.loadtxt("test/hypercube_X_test_std.csv", delimiter=',')
y_test = np.loadtxt("test/hypercube_y_test.csv", delimiter=',')

print('\nОбучаем модель случайного леса на наших данных...')

# Создаём экземпляр классификатора с количеством деревьев, равным 100
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

#Проверим, насколько хорошо модель описывает данные:
print('accuracy_score = ',accuracy_score(y_test, clf.predict(X_test)))

'''
Алгоритм случайного леса использует коэффициенты Джини для того, чтобы измерить, 
как снижается метрика accuracy модели при исключении определённого признака. 
Чем больше значение коэффициента Джини, тем признак более значимый.

Посмотрим на 10 наиболее важных признаков, которые посчитал случайный лес:
plt.figure(num=None, figsize=(20,8), dpi=80, facecolor='w', edgecolor='k')

feat_importances = pd.Series(clf.feature_importances_)
feat_importances.nlargest(10).plot(kind='barh')

Теперь мы можем установить порог для отбора признаков. Это может быть вещественное число. 
Если значимость признаков будет меньше установленного порога, то они будут отсеяны. 
Также можно выбрать конкретное число нужных нам признаков.
'''
print('Значения коэффициентов Джини модели случайного леса: ',clf.feature_importances_)

# Отберём 10 наиболее важных признаков:
# получим индексы нужных признаков
imp_features_idx = np.argsort(clf.feature_importances_)[:-11:-1] #.index
print('Отсортируем и отберём 10 наиболее важных признаков: ',imp_features_idx)

# Снова обучим модель случайного леса на данных с отобранными признаками:
# clf = RandomForestClassifier(n_estimators=100)
# clf.fit(X_train[:, imp_features_idx], y_train)

# print(accuracy_score(y_test, clf.predict(X_test[:, imp_features_idx])))

print('Сохраняем модель в файл: model/RFC_model.pkl')
with open('model/RFC_model.pkl','wb') as f:
    pickle.dump(clf,f)

