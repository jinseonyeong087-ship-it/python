from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 한글 폰트 설정 (폰트가 깨지지 않도록)
plt.rc('font', family='Malgun Gothic')  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

digits = load_digits()
X = digits.data
y = digits.target

pca = PCA(n_components=2)    #2차원 변경
X_pca = pca.fit_transform(X)

kmeans = KMeans(n_clusters=10, random_state=42)  # KMeans 클러스터링(3개 군집)
clusters = kmeans.fit_predict(X)  # 군집 예측
pca2 = PCA(n_components=2)  # 2차원으로 차원 축소
X_pca2 = pca2.fit_transform(X) 

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.legend(*scatter.legend_elements(), title="Digits")
plt.title("PCA on Digits Dataset")
plt.xlabel("PC1")
plt.ylabel("PC2")

plt.subplot(1, 2, 2)
plt.scatter(X_pca2[:, 0], X_pca2[:, 1], c=clusters, cmap='viridis')  # 군집 결과 시각화
plt.title('KMeans 클러스터링 (Digits 데이터셋)')  # 한글 제목
plt.xlabel('PCA 1')  # x축 이름
plt.ylabel('PCA 2')  # y축 이름
plt.show()