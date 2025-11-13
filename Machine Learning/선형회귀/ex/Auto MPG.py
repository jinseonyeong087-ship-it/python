import pandas as pd  # pandas 라이브러리를 pd라는 이름으로 불러옴 (데이터 처리용)
from sklearn.linear_model import LinearRegression  # 선형 회귀 모델을 불러옴
from sklearn.model_selection import train_test_split  # 데이터 분할 함수 불러옴
import matplotlib.pyplot as plt  # 그래프 그리기용 pyplot을 plt로 불러옴
import matplotlib  # matplotlib 전체를 불러옴 (폰트 설정 등)

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/mpg.csv")  # 온라인에서 mpg 데이터셋을 읽어옴
df = df.dropna()  # 결측값(NA)이 있는 행을 모두 제거함
X = df[['horsepower', 'weight', 'displacement']]  # 입력 변수(특성)로 마력, 무게, 배기량을 선택함
y = df['mpg']  # 목표 변수(타겟)로 연비(mpg)를 선택함

# 데이터 구성 확인
print('데이터 컬럼명:', df.columns.tolist())
print('데이터 상위 5개:\n', df.head())
print('결측치 개수:\n', df.isnull().sum())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 데이터를 8:2 비율로 훈련/테스트로 나눔
model = LinearRegression()  # 선형 회귀 모델 객체 생성
model.fit(X_train, y_train)  # 훈련 데이터로 모델 학습

# 예측값 계산
y_pred = model.predict(X_test)  # 테스트 데이터로 연비 예측

# 한글 폰트 설정 (Windows 기준)
matplotlib.rc('font', family='Malgun Gothic')  # 그래프에서 한글이 깨지지 않도록 폰트 설정
plt.rcParams['axes.unicode_minus'] = False  # 마이너스(-) 기호가 깨지지 않도록 설정

# 산점도와 회귀선 시각화
plt.figure(figsize=(8, 6))  # 그래프 크기 설정
plt.scatter(y_test, y_pred, alpha=0.7)  # 실제값과 예측값의 산점도 그림
plt.xlabel('실제 연비(mpg)')  # x축 라벨 설정
plt.ylabel('예측 연비(mpg)')  # y축 라벨 설정
plt.title('실제값 vs 예측값')  # 그래프 제목 설정
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 대각선(이상적 예측선) 그림
plt.show()  # 그래프 출력

print("R^2 score:", model.score(X_test, y_test))  # 테스트 데이터에 대한 결정계수(R^2) 점수 출력