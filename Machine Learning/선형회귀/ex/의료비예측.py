import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 로드
df = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# 데이터 확인
print("=== 데이터 기본 정보 ===")
print(f"데이터 크기: {df.shape}")
print("\n=== 데이터 처음 5행 ===")
print(df.head())
print("\n=== 데이터 타입 ===")
print(df.dtypes)
print("\n=== 결측치 확인 ===")
print(df.isnull().sum())
print("\n=== 기술통계 ===")
print(df.describe())

# 범주형 변수 처리
df = pd.get_dummies(df, drop_first=True)

# 특성과 타겟 분리
X = df.drop('charges', axis=1)
y = df['charges']

# 데이터 시각화
plt.figure(figsize=(15, 10))

# 1. 타겟 변수 분포
plt.subplot(2, 3, 1)
plt.hist(y, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('의료비 분포')
plt.xlabel('의료비')
plt.ylabel('빈도')

# 2. 나이와 의료비 관계
plt.subplot(2, 3, 2)
plt.scatter(df['age'], y, alpha=0.6, color='green')
plt.title('나이 vs 의료비')
plt.xlabel('나이')
plt.ylabel('의료비')

# 3. BMI와 의료비 관계
plt.subplot(2, 3, 3)
plt.scatter(df['bmi'], y, alpha=0.6, color='orange')
plt.title('BMI vs 의료비')
plt.xlabel('BMI')
plt.ylabel('의료비')

# 4. 흡연 여부에 따른 의료비 박스플롯
plt.subplot(2, 3, 4)
smoker_data = [y[df['smoker_yes'] == 1], y[df['smoker_yes'] == 0]]
plt.boxplot(smoker_data, labels=['흡연자', '비흡연자'])
plt.title('흡연 여부에 따른 의료비')
plt.ylabel('의료비')

# 5. 성별에 따른 의료비 박스플롯
plt.subplot(2, 3, 5)
gender_data = [y[df['sex_male'] == 1], y[df['sex_male'] == 0]]
plt.boxplot(gender_data, labels=['남성', '여성'])
plt.title('성별에 따른 의료비')
plt.ylabel('의료비')

# 6. 지역별 의료비 박스플롯
plt.subplot(2, 3, 6)
region_cols = [col for col in df.columns if 'region' in col]
region_data = []
region_labels = []
for col in region_cols:
    region_data.append(y[df[col] == 1])
    region_labels.append(col.replace('region_', ''))

plt.boxplot(region_data, labels=region_labels)
plt.title('지역별 의료비')
plt.ylabel('의료비')
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# 상관관계 히트맵
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('특성 간 상관관계')
plt.tight_layout()
plt.show()

# 모델 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# 모델 성능 평가
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

print("\n=== 모델 성능 ===")
print(f"훈련 데이터 R^2 점수: {train_score:.4f}")
print(f"테스트 데이터 R^2 점수: {test_score:.4f}")

# 예측 vs 실제값 시각화
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('실제 의료비')
plt.ylabel('예측 의료비')
plt.title('실제값 vs 예측값')
plt.grid(True, alpha=0.3)
plt.show()

# 특성 중요도 (계수)
feature_importance = pd.DataFrame({
    '특성': X.columns,
    '계수': model.coef_
})
feature_importance = feature_importance.sort_values('계수', key=abs, ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance['특성'], feature_importance['계수'])
plt.xlabel('계수 값')
plt.title('특성 중요도 (선형 회귀 계수)')
plt.tight_layout()
plt.show()

print("\n=== 특성 중요도 ===")
print(feature_importance)