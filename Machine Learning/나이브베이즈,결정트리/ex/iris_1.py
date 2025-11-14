# 1. 필요한 라이브러리 불러오기
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 2. 데이터셋 불러오기
iris = load_iris()
X = iris.data
y = iris.target

# 한글 폰트 설정 (폰트가 깨지지 않도록)
plt.rc('font', family='Malgun Gothic')  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 3. 학습용/테스트용 데이터 분할 (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. 모델 생성 및 학습
nb = GaussianNB()
tree = DecisionTreeClassifier(max_depth=3)
nb.fit(X_train, y_train)
tree.fit(X_train, y_train)

# 5. 예측
y_pred = nb.predict(X_test)
y_pred1 = tree.predict(X_test)


error = []

# 6. 성능 평가 (GaussianNB)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))
print("Accuracy:", accuracy_score(y_test, y_pred))

# 7. 성능 평가 (Decision Tree)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred1))
print("\nClassification Report:")
print(classification_report(y_test, y_pred1, target_names=iris.target_names))
print("Accuracy:", accuracy_score(y_test, y_pred1))

plt.figure(figsize=(12, 5))

# (왼쪽) GaussianNB 결과
plt.subplot(1, 2, 1)
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=y_pred, style=y_test, s=100)
plt.title('테스트 데이터 예측 (색: 예측, 모양: 실제)')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend(title='예측/실제')

# (오른쪽) Decision Tree 결과
plt.subplot(1, 2, 2)
print(classification_report(y_test, tree.predict(X_test)))  # 분류 리포트 출력
# plot_tree의 class_names는 list 타입이어야 하므로 list로 변환
plot_tree(tree, feature_names=iris.feature_names, class_names=list(iris.target_names), filled=True)  # 결정트리 시각화
plt.title('결정트리 시각화')  # 그래프 제목


plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5)) 

plt.subplot(1,2,1)
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Naive Bayes')

plt.subplot(1,2,2)
sns.heatmap(confusion_matrix(y_test, y_pred1), annot=True, fmt='d', cmap='Oranges')
plt.title('Decision Tree')

plt.tight_layout()
plt.show()