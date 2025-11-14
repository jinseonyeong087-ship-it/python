from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

digits = load_digits()
X, y = digits.data, digits.target

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
print(classification_report(y_test, model.predict(X_test)))

# 데이터 일부 확인
print('데이터 샘플 (X):')
print(X[:5])
print('레이블 샘플 (y):')
print(y[:5])

# 혼동 행렬 시각화 및 한글 폰트 설정
matplotlib.rc('font', family='Malgun Gothic')  # 윈도우 한글 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('예측값')
plt.ylabel('실제값')
plt.title('혼동 행렬')
plt.show()