# =============================
# 🧠 머신러닝 분류 성능 비교 템플릿
# =============================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# =============================
# 1️⃣ 데이터 불러오기
# =============================
# CSV, 내장 데이터 등 어떤 것이든 가능
df = pd.read_csv("your_dataset.csv")   # 예: titanic.csv, diabetes.csv 등
X = df.drop("target", axis=1)          # 독립 변수
y = df["target"]                       # 종속 변수

# 필요하다면 인코딩
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

# =============================
# 2️⃣ 데이터 전처리 및 분리
# =============================
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# 3️⃣ 여러 분류기 정의
# =============================
models = {
    "Naive Bayes": GaussianNB(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
}

# =============================
# 4️⃣ 성능 평가
# =============================
results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results.append({"Model": name, "Accuracy": acc})
    print(f"\n📘 [{name}]")
    print(classification_report(y_test, y_pred))

# =============================
# 5️⃣ 시각화 (성능 비교)
# =============================
results_df = pd.DataFrame(results)
sns.barplot(x="Model", y="Accuracy", data=results_df, palette="viridis")
plt.title("분류 모델별 정확도 비교")
plt.ylim(0, 1)
plt.show()
