from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm  # 한글 폰트 설정을 위해 추가

iris = load_iris()  # 붓꽃 데이터셋 로드
X, y = iris.data, iris.target  # 특성과 타겟 분리

# 데이터 샘플 확인
print('데이터 샘플 확인:')
print(X[:5])
print('타겟 샘플 확인:')
print(y[:5])

# 한글 폰트 설정 (폰트가 깨지지 않도록)
plt.rc('font', family='Malgun Gothic')  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # 데이터 분할

model = DecisionTreeClassifier(max_depth=3)  # 결정트리 분류기 생성 (최대 깊이 3)
model.fit(X_train, y_train)  # 모델 학습

print(classification_report(y_test, model.predict(X_test)))  # 분류 리포트 출력
plt.figure(figsize=(12, 6))  # 그래프 크기 설정
# plot_tree의 class_names는 list 타입이어야 하므로 list로 변환
plot_tree(model, feature_names=iris.feature_names, class_names=list(iris.target_names), filled=True)  # 결정트리 시각화
plt.title('결정트리 시각화')  # 그래프 제목
plt.show()  # 그래프 출력