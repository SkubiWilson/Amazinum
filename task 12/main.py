import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns

data = pd.read_csv("Iris.csv")

features = data.drop(columns=["Id", "Species"])

print(features.head())

sns.pairplot(data.drop(columns=["Id"]), hue="Species", diag_kind="hist", markers=["o", "s", "D"])
plt.suptitle("Парні графіки ознак за видами", y=1.02)
plt.show()


sclr = StandardScaler()
normal_features = sclr.fit_transform(features)
normal_dt = pd.DataFrame(normal_features, columns=features.columns)
print("\n", normal_dt.head())



inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(normal_dt)
    inertia.append(kmeans.inertia_)


plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Кількість кластерів')
plt.ylabel('Інерція (Sum of Squared Distances)')
plt.title('Метод Elbow')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(normal_dt)
data['Cluster'] = clusters

pca = PCA(n_components=2)
reduced = pca.fit_transform(normal_dt)
data['PCA1'] = reduced[:, 0]
data['PCA2'] = reduced[:, 1]
fig = px.scatter(data, x="PCA1", y="PCA2", color="Cluster", symbol="Species")
fig.show()

score = silhouette_score(normal_dt, clusters)
print(f"Score: {score:.3f}")


