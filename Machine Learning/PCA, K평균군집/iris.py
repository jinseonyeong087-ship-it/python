# 1. 라이브러리 불러오기
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 2. 데이터 로드
iris = load_iris()
X = iris.data  # 우리는 라벨 y를 사용하지 않음 (비지도학습)

# 3. KMeans 모델 생성 (k=3)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# 4. 클러스터 예측
y_kmeans = kmeans.predict(X)

# 5. 2D 시각화를 위해 PCA로 차원 축소
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 6. 시각화
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', marker='o', s=50)
centers = pca.transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centers')
plt.title("K-Means Clustering on Iris Dataset (PCA-Reduced)")
plt.legend()
plt.show()

