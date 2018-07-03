import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import IPython
import sklearn
from IPython.display import display
import mglearn
X, y = mglearn.datasets.make_forge()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(['class_0', 'class_1'], loc=3)
plt.xlabel("first_feature")
plt.ylabel("second_feature")
print("X.shape: {}".format(X.shape))

X, y = mglearn.datasets.make_wave(n_samples=40)
plt.plot(X, y, 'o')
plt.ylim(-3, 3)
plt.xlabel('feature')
plt.ylabel('target')
# plt.show()

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer_key() : \n{}".format(cancer.keys()))
print("data_shape : {}".format(cancer.data.shape))
print("sample_count by class:\n{}".format(
    {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
))
print("feature_name:\n{}".format(cancer.feature_names))

### 2.3.2 k-Nearest Neighbors
## k-최근접 이웃 분류

mglearn.plots.plot_knn_classification(n_neighbors=3)
# plt.show()

from sklearn.model_selection import train_test_split
X, y = mglearn.datasets.make_forge()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
print('test_set_prediction : {}'.format(clf.predict(X_test)))
print('test_set_accuracy : {:.2f}'.format(clf.score(X_test, y_test)))

## KNeighborsClassifier 분석
fig, axes = plt.subplots(1, 3, figsize=(10, 3))

for n_neighbors, ax in zip([1, 3, 9], axes):
    # fit method는 self object를 반환한다.
    # 그래서 객체 생성과 fit method를 한 줄에 쓸 수 있다.
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
    mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.4)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbors".format(n_neighbors))
    ax.set_xlabel("feature0")
    ax.set_ylabel("feature1")
axes[0].legend(loc=3)
plt.show()

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target, random_state=66
)
training_accuracy = []
test_accuracy = []
neighbors_setting = range(1, 11)

for n_neighbors in neighbors_setting:
    # model create
    clf = KNeighborsClassifier(n_neighbors)
    clf.fit(X_train, y_train)
    # training dataset accuracy save
    training_accuracy.append(clf.score(X_train, y_train))
    # generalization accuracy save(test dataset accuracy)
    test_accuracy.append(clf.score(X_test, y_test))

plt.plot(neighbors_setting, training_accuracy, label='training accuracy')
plt.plot(neighbors_setting, test_accuracy, label='test accyracy')
plt.ylabel('accuracy')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()

## k-nearest neighbors regression
mglearn.plots.plot_knn_regression(1)
mglearn.plots.plot_knn_regression(3)
plt.show()

from sklearn.neighbors import KNeighborsRegressor

X, y = mglearn.datasets.make_wave(n_samples=40)
# wave dataset을 training set와 test set로 나눈다.
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# neighbors number를 3으로 하여 model의 object를 만든다.
reg = KNeighborsRegressor(n_neighbors=3)
# training data와 target data를 사용하여 model을 학습시킨다.
reg.fit(X_train, y_train)

print('test set prediction:\n{}'.format(reg.predict(X_test)))
print('test set R^2:{:.2f}'.format(reg.score(X_test, y_test)))