# 1. 라이브러리 불러오기
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 2. 데이터 로드
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# 3. PCA 모델 생성 (2차원으로 축소)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 4. K-means Clustering 적용 (PCA 결과에 대해)
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X_pca)

# 5. 시각화를 위한 데이터프레임 구성
df_pca = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
df_pca['target'] = y
df_pca['cluster'] = y_kmeans

# 6. 시각화 - 원본 라벨 기준
plt.figure(figsize=(16, 6))

# 서브플롯 1: 원본 라벨 기준
plt.subplot(1, 2, 1)
for i, name in enumerate(target_names):
    plt.scatter(df_pca[df_pca['target'] == i]['PC1'],
                df_pca[df_pca['target'] == i]['PC2'],
                label=name)

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of IRIS Dataset (Original Labels)')
plt.legend()
plt.grid(True)

# 서브플롯 2: K-means 클러스터링 결과
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='viridis', marker='o', s=50)
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X', s=200, label='Centers')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA with K-Means Clustering (k=3)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 7. 클러스터링 결과 출력
print("\n=== K-Means Clustering Results ===")
print(f"클러스터 중심점:\n{centers}")
print(f"\n각 데이터 포인트의 클러스터 할당:")
print(df_pca[['PC1', 'PC2', 'cluster']].head(10))

