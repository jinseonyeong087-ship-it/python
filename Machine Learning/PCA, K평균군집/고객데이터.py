import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 한글 폰트 설정 (폰트가 깨지지 않도록)
plt.rc('font', family='Malgun Gothic')  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

df = pd.DataFrame({
    'Age': [23, 45, 31, 35, 40, 22, 29, 60, 55, 47],
    'SpendingScore': [77, 40, 60, 65, 35, 90, 75, 20, 25, 30],
    'Income': [30, 80, 50, 60, 70, 25, 45, 100, 90, 85]
})

X_scaled = StandardScaler().fit_transform(df)
X_pca = PCA(n_components=2).fit_transform(X_scaled)

kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1])
plt.title("Customer Data PCA")
plt.xlabel("PC1")
plt.ylabel("PC2")


plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis')  # 군집 결과 시각화
plt.title('KMeans 클러스터링 (Customer 데이터셋)')  # 한글 제목
plt.xlabel('PCA 1')  # x축 이름
plt.ylabel('PCA 2')  # y축 이름
plt.show()