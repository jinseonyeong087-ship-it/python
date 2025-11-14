# ===============================
# 🔢 KNN - K 값 변화 시각화 예제
# ===============================

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ✅ 한글 폰트 설정 (Windows: 맑은 고딕)
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지


# 1️⃣ 데이터 불러오기
digits = load_digits()
X, y = digits.data, digits.target

# 2️⃣ 표준화 (정규화)
X = StandardScaler().fit_transform(X)

# 3️⃣ 학습 / 테스트 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4️⃣ 여러 K값을 시험해보기
k_values = range(1, 16)   # K = 1 ~ 15
accuracies = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)  # 정확도 계산
    accuracies.append(score)
    print(f"K={k} -> 정확도: {score:.4f}")

# 5️⃣ 시각화
plt.figure(figsize=(8, 5))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='navy')
plt.title('K값 변화에 따른 KNN 정확도')
plt.xlabel('K (이웃 개수)')
plt.ylabel('정확도 (Accuracy)')
plt.xticks(k_values)
plt.grid(True)
plt.show()
