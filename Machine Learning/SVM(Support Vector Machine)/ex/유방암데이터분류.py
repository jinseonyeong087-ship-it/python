# 1. 라이브러리 불러오기
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 2. 데이터 불러오기
data = load_breast_cancer()
X = data.data
y = data.target
print("Target labels:", data.target_names)  # ['malignant' 'benign']

# 데이터 확인을 위해 추가
print("\n--- 데이터 샘플 확인 ---")
print("특성 데이터 (X) 상위 5개:\n", X[:5])
print("타겟 데이터 (y) 상위 5개:\n", y[:5])
print("특성 이름:\n", data.feature_names)

# 3. 데이터 전처리 (스케일링)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. 훈련/테스트 분할
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. SVM 모델 학습
svm = SVC(kernel='linear', C=1.0)  # 선형 커널 SVM
svm.fit(X_train, y_train)

# 6. 예측
y_pred = svm.predict(X_test)

# 7. 평가 결과 출력
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 8. Confusion Matrix 시각화
plt.rcParams['font.family'] = 'NanumGothicCoding' # 한글 폰트 설정 (이전에 설치된 폰트 사용)
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

fig, ax = plt.subplots(figsize=(6, 6))
display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred),
                                 display_labels=data.target_names)
display.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_title('Confusion Matrix')
plt.show()