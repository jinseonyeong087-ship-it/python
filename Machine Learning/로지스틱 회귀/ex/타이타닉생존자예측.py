import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import platform
from sklearn.metrics import confusion_matrix
import matplotlib

# 한글 폰트 설정 (운영체제별)
if platform.system() == 'Windows':
    matplotlib.rc('font', family='Malgun Gothic')
else:
    matplotlib.rc('font', family='AppleGothic')
matplotlib.rcParams['axes.unicode_minus'] = False

df = sns.load_dataset("titanic").dropna(subset=["age", "fare", "embarked", "sex"])
df = df[df['embarked'].notnull()]
df['sex'] = df['sex'].map({'male': 0, 'female': 1})
df['embarked'] = df['embarked'].map({'S': 0, 'C': 1, 'Q': 2})

X = df[['age', 'fare', 'sex', 'embarked']]
y = df['survived']

X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

# 데이터 일부 확인
print('데이터 샘플:')
print(df.head())

print(classification_report(y_test, model.predict(X_test)))

# Confusion Matrix 시각화
cm = confusion_matrix(y_test, model.predict(X_test))
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['사망', '생존'], yticklabels=['사망', '생존'])
plt.xlabel('예측값')
plt.ylabel('실제값')
plt.title('Confusion Matrix')
plt.show()