# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import DBSCAN 
from sklearn.datasets import make_moons 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import PCA 

# %%
blob_centers = np.array(
    [[0.2, 2.3], [-1.5, 2.3], [-2.8, 1.8], [-2.8, 2.8], [-2.8, 1.3]]
)
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

# %%
X, y = make_blobs(
    n_samples=2000, centers=blob_centers, cluster_std=blob_std, random_state=7
)

# %%
k = 5
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(X)

# %%
y_pred

# %%
kmeans.cluster_centers_

# %%
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
kmeans.predict(X_new)

# %%
kmeans.transform(X_new)

# %%
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)

# %%
minibatch_kmeans = MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)

# %%
silhouette_score(X, minibatch_kmeans.labels_)

# %%
X_digits, y_digits = load_digits(return_X_y=True)

# %%
X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)


# %%
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# %%
log_reg.score(X_test, y_test)

# %%
pipeline = Pipeline(
    [("kmeans", KMeans(n_clusters=50)), ("log_reg", LogisticRegression()), ]
)
pipeline.fit(X_train, y_train)

# %%
pipeline.score(X_test, y_test)

# %%
param_grid = dict(kmeans__n_clusters=range(2, 100))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
grid_clf.fit(X_train, y_train)

# %%
grid_clf.best_params_

# %%
grid_clf.score(X_test, y_test)

# %%
n_labeled = 50
log_reg = LogisticRegression()
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])

# %%
log_reg.score(X_test, y_test)

# %%
k = 50
kmeans = KMeans(n_clusters=k)
X_digits_dist = kmeans.fit_transform(X_train)
representative_digit_idx = np.argmin(X_digits_dist, axis=0)
X_representative_digits = X_train[representative_digit_idx]

# %%
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(
        X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear"
    )
    plt.axis("off")

plt.show()
# %%
y_representative_digits = np.array(
    [
        4,
        3,
        6,
        5,
        7,
        0,
        2,
        8,
        5,
        1,
        6,
        4,
        9,
        2,
        9,
        5,
        7,
        1,
        8,
        2,
        5,
        2,
        3,
        6,
        8,
        1,
        5,
        8,
        3,
        8,
        7,
        8,
        9,
        8,
        3,
        4,
        7,
        0,
        8,
        4,
        7,
        7,
        6,
        2,
        2,
        5,
        4,
        0,
        3,
        5,
    ]
)

# %%
log_reg = LogisticRegression(
    multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42
)
log_reg.fit(X_representative_digits, y_representative_digits)
log_reg.score(X_test, y_test)

# %%
y_train_propagated = np.empty(len(X_train), dtype=np.int32)
for i in range(k):
    y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

# %%
log_reg = LogisticRegression(
    multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42
)
log_reg.fit(X_train, y_train_propagated)

# %%
log_reg.score(X_test, y_test)

# %%
percentile_closest = 20

X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
for i in range(k):
    in_cluster = kmeans.labels_ == i
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = X_cluster_dist > cutoff_distance
    X_cluster_dist[in_cluster & above_cutoff] = -1

# %%
partially_propagated = X_cluster_dist != -1
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]

# %%
log_reg = LogisticRegression(
    multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42
)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)

# %%
log_reg.score(X_test, y_test)

# %%
np.mean(y_train_partially_propagated == y_train[partially_propagated])

# %%
X, y = make_moons(n_samples=1000, noise=.05)
dbscan = DBSCAN(eps=.05, min_samples=5)
dbscan.fit(X)

# %%
dbscan.labels_

# %%
len(dbscan.core_sample_indices_)

# %%
knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

# %%
X_new = np.array([[-.05, 0], [0, .5], [1, -.01], [2, 1]])
knn.predict(X_new)
knn.predict_proba(X_new)
# %%
y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
y_pred[y_dist > .2] = -1
y_pred.ravel()

# %%
gm = GaussianMixture(n_components=3, n_init=10)
gm.fit(X)

# %%
gm.weights_ 
gm.means_
gm.covariances_

# %%
gm.converged_
gm.n_iter_

# %%
gm.predict(X)
gm.predict_proba(X)

# %%
X_new, y_new = gm.sample(6)
X_new 
y_new

# %%
gm.score_samples(X)

# %%
densities = gm.score_samples(X)
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]

# %%
gm.bic(X)

# %%
gm.aic(X)

# %%
bgm = BayesianGaussianMixture(n_components=10, n_init=10)
bgm.fit(X)
np.round(bgm.weights_, 2)

# %%
olivetti = fetch_olivetti_faces()
print(olivetti.DESCR)

# %%
olivetti.target

# %%
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=40, random_state=42)
train_valid_idx, test_idx = next(strat_split.split(olivetti.data, olivetti.target)) 
X_train_valid = olivetti.data[train_valid_idx]
y_train_valid = olivetti.target[train_valid_idx]
X_test = olivetti.data[test_idx]
y_test = olivetti.target[test_idx]

strat_split = StratifiedShuffleSplit(n_splits=1, test_size=80, random_state=43)
train_idx, valid_idx = next(strat_split.split(X_train_valid, y_train_valid)) 
X_train = X_train_valid[train_idx]
y_train = y_train_valid[train_idx]
X_valid = X_train_valid[valid_idx]
y_valid = y_train_valid[valid_idx]

# %%
