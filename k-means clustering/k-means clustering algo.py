import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
df = pd.read_csv('Wholesale.csv')
df = pd.get_dummies(df, columns=['Channel', 'Region'])
scaled = MinMaxScaler().fit_transform(df)
inertia = []
for k in range(1, 11):
    model = KMeans(n_clusters=k)
    model.fit(scaled)
    inertia.append(model.inertia_)
plt.plot(range(1, 11), inertia, 'bx-')
plt.title("Elbow Method")
plt.xlabel("k")
plt.ylabel("Inertia")
plt.grid(True)
plt.show()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
n_components = 2
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X_scaled)
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis', edgecolor='k',
s=100)
plt.colorbar(scatter, label='Target Label')
plt.title('Data in Reduced-Dimensional Space (2 Components)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.show()
