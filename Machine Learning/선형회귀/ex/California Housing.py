from sklearn.datasets import fetch_california_housing  # 캘리포니아 주택 가격 데이터셋을 불러오는 함수
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델 클래스
from sklearn.model_selection import train_test_split  # 데이터셋을 학습/테스트 세트로 분할하는 함수
import matplotlib.pyplot as plt  # 데이터 시각화를 위한 라이브러리
import matplotlib  # matplotlib 설정을 위한 라이브러리
matplotlib.rc('font', family='Malgun Gothic')  # 한글 폰트 설정(맑은 고딕)
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 기호가 깨지지 않도록 설정

data = fetch_california_housing()  # 캘리포니아 주택 가격 데이터셋 로드

# 데이터셋 정보 출력
print('특성 이름:', data.feature_names)
print('데이터 샘플(5개):')
print(data.data[:5])

X = data.data  # 입력 변수(특성) 데이터
y = data.target  # 타겟(주택 가격) 데이터

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 데이터를 8:2 비율로 학습/테스트 세트로 분할

model = LinearRegression()  # 선형 회귀 모델 객체 생성
model.fit(X_train, y_train)  # 학습 데이터로 모델 학습

print("R^2 score:", model.score(X_test, y_test))  # 테스트 세트에 대한 R^2 점수(모델 성능) 출력

y_pred = model.predict(X_test)  # 테스트 세트에 대한 예측값 생성

plt.figure(figsize=(8, 6))  # 그래프 크기 설정
plt.scatter(y_test, y_pred, alpha=0.5)  # 실제 값과 예측 값의 산점도 그리기
plt.xlabel('실제 값')  # x축 레이블 설정
plt.ylabel('예측 값')  # y축 레이블 설정
plt.title('실제 값 vs 예측 값')  # 그래프 제목 설정
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 대각선(이상적인 예측선) 추가
plt.show()  # 그래프 출력