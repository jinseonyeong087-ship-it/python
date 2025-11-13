import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

df = pd.read_csv("https://raw.githubusercontent.com/arunk13/MSDA-Assignments/master/IS607Fall2015/Assignment3/student-mat.csv", sep=";")
X = df[['studytime', 'failures', 'absences', 'G1', 'G2']]
y = df['G3']  # Final grade

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = LinearRegression()
model.fit(X_train, y_train)

print("R^2 score:", model.score(X_test, y_test))

print("데이터 샘플:")
print(df.head())

# 한글 폰트 설정 (Windows의 경우 보통 'Malgun Gothic')
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 예측값 생성
y_pred = model.predict(X_test)

# 실제값과 예측값 시각화
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.xlabel('실제 G3')
plt.ylabel('예측 G3')
plt.title('실제값 vs 예측값')
# y_test가 Series가 아닐 경우를 대비해 numpy array로 변환
y_test_np = np.array(y_test)
plt.plot([y_test_np.min(), y_test_np.max()], [y_test_np.min(), y_test_np.max()], 'r--')
plt.show()

