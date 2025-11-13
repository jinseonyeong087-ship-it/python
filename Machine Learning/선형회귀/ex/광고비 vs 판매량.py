import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 컬럼명을 직접 지정
column_names = ['Index', 'TV', 'Radio', 'Newspaper', 'Sales']
url = "https://www.statlearning.com/s/Advertising.csv"
df = pd.read_csv(url, header=None, names=column_names, skiprows=1)

# Index 컬럼은 불필요하므로 제거
df = df.drop(columns=['Index'])

# 데이터 확인
print("=== 데이터 기본 정보 ===")
print(f"데이터 형태: {df.shape}")
print(f"컬럼명: {list(df.columns)}")
print("\n=== 데이터 처음 5행 ===")
print(df.head())
print("\n=== 데이터 통계 정보 ===")
print(df.describe())
print("\n=== 결측치 확인 ===")
print(df.isnull().sum())

# 설명 변수와 타겟 지정
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# 데이터 분할, 모델 학습 및 평가
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print("\n=== 모델 성능 ===")
print("R² score:", model.score(X_test, y_test))

# 시각화
plt.figure(figsize=(15, 10))

# 1. 각 변수의 분포 확인
plt.subplot(2, 3, 1)
plt.hist(df['TV'], bins=20, alpha=0.7, color='skyblue')
plt.title('TV 광고 비용 분포')
plt.xlabel('TV 광고 비용')
plt.ylabel('빈도')

plt.subplot(2, 3, 2)
plt.hist(df['Radio'], bins=20, alpha=0.7, color='lightgreen')
plt.title('Radio 광고 비용 분포')
plt.xlabel('Radio 광고 비용')
plt.ylabel('빈도')

plt.subplot(2, 3, 3)
plt.hist(df['Newspaper'], bins=20, alpha=0.7, color='salmon')
plt.title('Newspaper 광고 비용 분포')
plt.xlabel('Newspaper 광고 비용')
plt.ylabel('빈도')

# 2. 각 변수와 Sales의 관계
plt.subplot(2, 3, 4)
plt.scatter(df['TV'], df['Sales'], alpha=0.6, color='skyblue')
plt.title('TV 광고 비용 vs Sales')
plt.xlabel('TV 광고 비용')
plt.ylabel('Sales')

plt.subplot(2, 3, 5)
plt.scatter(df['Radio'], df['Sales'], alpha=0.6, color='lightgreen')
plt.title('Radio 광고 비용 vs Sales')
plt.xlabel('Radio 광고 비용')
plt.ylabel('Sales')

plt.subplot(2, 3, 6)
plt.scatter(df['Newspaper'], df['Sales'], alpha=0.6, color='salmon')
plt.title('Newspaper 광고 비용 vs Sales')
plt.xlabel('Newspaper 광고 비용')
plt.ylabel('Sales')

plt.tight_layout()
plt.show()

# 상관관계 히트맵
plt.figure(figsize=(8, 6))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('변수 간 상관관계')
plt.tight_layout()
plt.show()

# 모델 계수 출력
print("\n=== 선형 회귀 모델 계수 ===")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef:.4f}")
print(f"절편: {model.intercept_:.4f}")

y_pred = model.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # 대각선
plt.xlabel("실제 Sales")
plt.ylabel("예측된 Sales")
plt.title("실제값 vs 예측값")
plt.grid(True)
plt.tight_layout()
plt.show()