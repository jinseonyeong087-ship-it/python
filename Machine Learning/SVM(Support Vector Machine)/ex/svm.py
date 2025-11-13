import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV

# 🔹 현재 파일(svm.py)의 절대경로를 기준으로 file/svm 폴더 경로 계산
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "../../../file/svm"))

# 색상 정의 (빨강, 파랑)
red_RGB = (1, 0, 0)
blue_RGB = (0, 0, 1)
data_colors = [red_RGB, blue_RGB]


# --------------------------
# (1) 텍스트 파일에서 좌표 데이터 읽기
# --------------------------
def read_points_file(filename):
    points = []
    with open(filename, "r") as f:
        for point in f:
            point = point.strip("\n").split()
            points.append([float(point[0]), float(point[1])])
    return points


# --------------------------
# (2) 두 개의 클래스 데이터 파일을 읽어 결합
# --------------------------
def read_data(class_0_file, class_1_file):
    points_label0 = read_points_file(class_0_file)
    points_label1 = read_points_file(class_1_file)
    points = np.array(points_label0 + points_label1)
    labels = [0] * len(points_label0) + [1] * len(points_label1)
    return (points, labels)


# --------------------------
# (3) 학습/테스트 데이터 시각화
# --------------------------
def plot_data(X_train, y_train, X_test, y_test):
    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    colors = get_colors(y)
    colors_train = get_colors(y_train)
    colors_test = get_colors(y_test)

    plt.figure(figsize=(12, 4), dpi=150)

    plt.subplot(131)
    plt.axis('equal')
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=10, edgecolors=colors)
    plt.title("Data (100%)")

    plt.subplot(132)
    plt.axis('equal')
    plt.scatter(X_train[:, 0], X_train[:, 1], c=colors_train, s=10, edgecolors=colors_train)
    plt.title("Training Data (80%)")

    plt.subplot(133)
    plt.axis('equal')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors_test, s=10, edgecolors=colors_test)
    plt.title("Test Data (20%)")

    plt.tight_layout()
    plt.show()


def get_colors(y):
    return [data_colors[label] for label in y]


def plot_decision_function(X_train, y_train, X_test, y_test, clf):
    plt.figure(figsize=(8, 4), dpi=150)

    plt.subplot(121)
    plt.title("Training data")
    plot_decision_function_helper(X_train, y_train, clf)

    plt.subplot(122)
    plt.title("Test data")
    plot_decision_function_helper(X_test, y_test, clf, True)
    plt.show()


def plot_decision_function_helper(X, y, clf, show_only_decision_function=False):
    colors = get_colors(y)
    plt.axis('equal')
    plt.tight_layout()
    plt.scatter(X[:, 0], X[:, 1], c=colors, s=10, edgecolors=colors)
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = clf.decision_function(xy).reshape(XX.shape)

    if show_only_decision_function:
        ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])
    else:
        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
                   alpha=0.5, linestyles=['--', '-', '--'])


# =========================================================
#  1️⃣ Linear SVM (선형 분리 가능 데이터)
# =========================================================
x, labels = read_data(
    os.path.join(DATA_DIR, "points_class_0.txt"),
    os.path.join(DATA_DIR, "points_class_1.txt")
)
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=0)

print("Displaying data. Close window to continue.")
plot_data(X_train, y_train, X_test, y_test)

clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

print("Displaying decision function. Close window to continue.")
plot_decision_function(X_train, y_train, X_test, y_test, clf)
print("Accuracy: {}%".format(clf.score(X_test, y_test) * 100))


# =========================================================
#  2️⃣ Linear SVM (노이즈 데이터)
# =========================================================
x, labels = read_data(
    os.path.join(DATA_DIR, "points_class_0_noise.txt"),
    os.path.join(DATA_DIR, "points_class_1_noise.txt")
)
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=0)

print("Displaying data. Close window to continue.")
plot_data(X_train, y_train, X_test, y_test)

clf_1 = svm.SVC(kernel='linear', C=1)
clf_1.fit(X_train, y_train)
print("Display decision function (C=1)...")
plot_decision_function(X_train, y_train, X_test, y_test, clf_1)

clf_100 = svm.SVC(kernel='linear', C=100)
clf_100.fit(X_train, y_train)

print("Accuracy(C=1): {}%".format(clf_1.score(X_test, y_test) * 100))
print("Display decision function (C=100)...")
plot_decision_function(X_train, y_train, X_test, y_test, clf_100)
print("Accuracy(C=100): {}%".format(clf_100.score(X_test, y_test) * 100))


# =========================================================
#  3️⃣ Non-linear SVM (비선형 데이터)
# =========================================================
x, labels = read_data(
    os.path.join(DATA_DIR, "points_class_0_nonLinear.txt"),
    os.path.join(DATA_DIR, "points_class_1_nonLinear.txt")
)
X_train, X_test, y_train, y_test = train_test_split(x, labels, test_size=0.2, random_state=0)

print("Displaying data.")
plot_data(X_train, y_train, X_test, y_test)

print("Training SVM ...")
clf = svm.SVC(C=10.0, kernel='rbf', gamma=0.1)
clf.fit(X_train, y_train)

print("Displaying decision function.")
plot_decision_function(X_train, y_train, X_test, y_test, clf)


# =========================================================
#  4️⃣ Grid Search (최적 파라미터 찾기)
# =========================================================
print("Performing grid search ... ")

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001, 0.00001, 10]
}

clf_grid = GridSearchCV(svm.SVC(), param_grid, verbose=1)
clf_grid.fit(X_train, y_train)

print("Best Parameters:\n", clf_grid.best_params_)
print("Best Estimators:\n", clf_grid.best_estimator_)

print("Displaying decision function for best estimator.")
plot_decision_function(X_train, y_train, X_test, y_test, clf_grid)
