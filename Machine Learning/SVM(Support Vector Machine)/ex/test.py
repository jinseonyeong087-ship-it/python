import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_moons

# 🌙 반달 모양 데이터(비선형 구조) 생성
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

# 두 가지 SVM 모델 생성
svm_linear = svm.SVC(kernel='linear', C=1.0)
svm_rbf = svm.SVC(kernel='rbf', C=1.0, gamma=0.5)

# 학습
svm_linear.fit(X, y)
svm_rbf.fit(X, y)

# 시각화를 위한 격자 생성
xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 300),
                     np.linspace(-1.0, 1.5, 300))

# 각 모델의 예측 결과 (결정경계용)
Z_linear = svm_linear.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_rbf = svm_rbf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z_linear = Z_linear.reshape(xx.shape)
Z_rbf = Z_rbf.reshape(xx.shape)

# ===== 그래프 시각화 =====
plt.figure(figsize=(12, 5))

# (1) Linear Kernel
plt.subplot(1, 2, 1)
plt.contourf(xx, yy, Z_linear > 0, alpha=0.3, cmap=plt.cm.coolwarm)
plt.contour(xx, yy, Z_linear, levels=[0], linewidths=2, colors='k')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("SVM - Linear Kernel (직선 경계)")

# (2) RBF Kernel
plt.subplot(1, 2, 2)
plt.contourf(xx, yy, Z_rbf > 0, alpha=0.3, cmap=plt.cm.coolwarm)
plt.contour(xx, yy, Z_rbf, levels=[0], linewidths=2, colors='k')
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
plt.title("SVM - RBF Kernel (비선형 곡선 경계)")

plt.tight_layout()
plt.show()
