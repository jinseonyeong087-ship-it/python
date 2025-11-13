from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib

# 한글 폰트 설정 (윈도우 기본 폰트 사용)
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=200)
model.fit(X_train, y_train)

# 데이터프레임으로 변환하여 데이터 확인
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['target'] = y
print('데이터 샘플:')
print(iris_df.head())

print(classification_report(y_test, model.predict(X_test)))

# 예측 결과 시각화
pred = model.predict(X_test)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_test[:, 0], y=X_test[:, 1], hue=pred, style=y_test, palette='Set1', s=100)
plt.title('테스트 데이터 예측 결과 (색: 예측, 모양: 실제)')
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.legend(title='예측/실제')
plt.show()