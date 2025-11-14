# 1️⃣ 라이브러리 불러오기
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree as sk_plot_tree
from xgboost import XGBClassifier, plot_tree as xgb_plot_tree

# 2️⃣ 한글 폰트 설정
plt.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False

# 3️⃣ 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target

# 4️⃣ 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5️⃣ 모델 생성
forest = RandomForestClassifier(n_estimators=3, random_state=42)
xg = XGBClassifier(
    n_estimators=3,      # 트리 3개만 (보기 쉽게)
    learning_rate=0.1,
    max_depth=3,
    random_state=42,
    eval_metric='mlogloss'
)

# 6️⃣ 학습
forest.fit(X_train, y_train)
xg.fit(X_train, y_train)

# 7️⃣ 랜덤포레스트 트리 시각화
plt.figure(figsize=(20, 8))
for i, estimator in enumerate(forest.estimators_):
    plt.subplot(1, len(forest.estimators_), i + 1)
    sk_plot_tree(
        estimator,
        feature_names=iris.feature_names,
        class_names=list(iris.target_names),
        filled=True,
        fontsize=8
    )
    plt.title(f"랜덤포레스트 트리 {i + 1}")

plt.suptitle("랜덤포레스트 구성 트리 시각화", fontsize=16)
plt.tight_layout()
plt.show()

# 8️⃣ XGBoost 트리 시각화
plt.figure(figsize=(25, 10))
for i in range(3):  # 0~2번 트리만 시각화
    plt.subplot(1, 3, i + 1)
    xgb_plot_tree(xg, num_trees=i, rankdir='LR')  # 👈 XGBoost 전용 함수
    plt.title(f"XGBoost 트리 {i+1}")

plt.suptitle("XGBoost 트리 시각화", fontsize=18, y=1.02)
plt.tight_layout()
plt.show()
