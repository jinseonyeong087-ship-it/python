from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# 한글 폰트 설정
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 1. 데이터 로드
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. 모델 학습
nb = GaussianNB()
tree = DecisionTreeClassifier(max_depth=3, random_state=42)
nb.fit(X_train, y_train)
tree.fit(X_train, y_train)

# 3. 예측
y_pred_nb = nb.predict(X_test)
y_pred_tree = tree.predict(X_test)

# 4. 시각화 (한 그래프에 두 모델)
plt.figure(figsize=(8,6))

# Naive Bayes 결과 (파란 원)
plt.scatter(X_test[:, 2], X_test[:, 3],
            c=y_pred_nb, cmap='cool', s=80, edgecolor='blue',
            alpha=0.6, marker='o')

# Decision Tree 결과 (빨간 X)
plt.scatter(X_test[:, 2], X_test[:, 3],
            c=y_pred_tree, cmap='autumn', s=50, edgecolor='red',
            alpha=0.9, marker='x')

plt.xlabel('꽃잎 길이 (petal length)')
plt.ylabel('꽃잎 너비 (petal width)')
plt.title('Naive Bayes(○) vs Decision Tree(X) 예측 비교')
plt.grid(True)
plt.show()
