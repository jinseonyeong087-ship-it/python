import pandas as pd  # pandas 라이브러리 불러오기
from sklearn.model_selection import train_test_split  # 데이터 분할 함수 불러오기
from sklearn.feature_extraction.text import TfidfVectorizer  # TF-IDF 벡터화 도구 불러오기
from sklearn.svm import SVC  # SVM 분류기 불러오기
from sklearn.metrics import classification_report, confusion_matrix  # 평가 지표 함수 불러오기
import matplotlib.pyplot as plt  # matplotlib 시각화 라이브러리 불러오기
import seaborn as sns  # seaborn 시각화 라이브러리 불러오기

# 1. 데이터 불러오기 및 전처리
# 데이터셋을 인터넷에서 불러오고, 컬럼명을 지정함
# 데이터프레임의 'label' 컬럼을 문자에서 숫자로 변환(ham:0, spam:1)
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv", sep='\t', names=['label', 'message'])
print(df.head())
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # 문자 라벨을 숫자로 매핑

# 2. 학습/테스트 분할
# 메시지와 라벨을 학습/테스트 데이터로 분할(test_size=0.2는 20%를 테스트로)
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 3. TF-IDF 벡터화
# 텍스트 데이터를 TF-IDF 벡터로 변환
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)  # 학습 데이터에 대해 벡터화 학습 및 변환
X_test_vec = vectorizer.transform(X_test)  # 테스트 데이터는 변환만 수행

# 4. 모델 학습
# SVM(선형 커널) 모델 생성 및 학습
model = SVC(kernel='linear')
model.fit(X_train_vec, y_train)

# 5. 예측
# 테스트 데이터에 대해 예측 수행
y_pred = model.predict(X_test_vec)

# 6. classification_report 출력
# 정밀도, 재현율, F1 점수 등 평가 지표 출력
print("정밀도, 재현율, F1 점수 등 평가 지표 출력")
print(classification_report(y_test, y_pred, target_names=['ham', 'spam']))

# ====== 시각화 코드 시작 ======

# 1. 혼동 행렬 시각화
cm = confusion_matrix(y_test, y_pred)  # 혼동 행렬 계산
plt.figure(figsize=(5, 4))  # 그래프 크기 설정
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['ham', 'spam'], yticklabels=['ham', 'spam'])  # 히트맵 그리기
plt.xlabel('Predicted')  # x축 라벨
plt.ylabel('Actual')  # y축 라벨
plt.title('Confusion Matrix')  # 그래프 제목
plt.show()  # 그래프 출력

# 2. classification_report 시각화
report = classification_report(y_test, y_pred, output_dict=True)  # 평가 지표를 딕셔너리로 반환

metrics = ['precision', 'recall', 'f1-score']  # 사용할 평가 지표 목록
labels = ['ham', 'spam']  # 라벨 이름
raw_labels = ['0', '1']  # 내부적으로 사용되는 라벨 키

# 점수 추출
scores = {
    metric: [report[label][metric] for label in raw_labels]
    for metric in metrics
}

x = range(len(labels))  # x축 위치
width = 0.2  # 막대 너비

plt.figure(figsize=(8, 5))  # 그래프 크기 설정
for i, metric in enumerate(metrics):
    plt.bar([p + width * i for p in x], scores[metric], width=width, label=metric)  # 각 지표별 막대그래프

plt.xticks([p + width for p in x], labels)  # x축 라벨 설정
plt.ylim(0, 1.1)  # y축 범위 설정
plt.ylabel('Score')  # y축 라벨
plt.title('Classification Report Metrics')  # 그래프 제목
plt.legend()  # 범례 표시
plt.show()  # 그래프 출력
# ====== 시각화 코드 끝 ======