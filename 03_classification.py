# %%
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

# %%
np.random.seed(42)
mpl.rc("axes", labelsize=14)
mpl.rc("xtick", labelsize=12)
mpl.rc("ytick", labelsize=12)

# %%
mnist = fetch_openml("mnist_784", version=1)
mnist.keys()

# %%
X, y = mnist["data"], mnist["target"]
X.shape

# %%
y.shape

# %%
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()

# %%
y[0]

# %%
y = y.astype(np.uint8)

# %%
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# %%
y_train_5 = y_train == 5
y_test_5 = y_test == 5

# %%
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# %%
sgd_clf.predict([some_digit])

# %%
cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %%


class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


# %%
never_5_clf = Never5Classifier()
cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

# %%
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# %%
confusion_matrix(y_train_5, y_train_pred)

# %%
y_train_perfect_predictions = y_train_5
confusion_matrix(y_train_5, y_train_perfect_predictions)

# %%
precision_score(y_train_5, y_train_pred)

# %%
recall_score(y_train_5, y_train_pred)

# %%
f1_score(y_train_5, y_train_pred)

# %%
y_scores = sgd_clf.decision_function([some_digit])
y_scores

# %%
threshold = 0
y_some_digit_pred = y_scores > threshold

# %%
threshold = 8000
y_some_digit_pred = y_scores > threshold
y_some_digit_pred

# %%
y_scores = cross_val_predict(
    sgd_clf, X_train, y_train_5, cv=3, method="decision_function"
)

# %%
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# %%


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1])


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# %%
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

# %%
y_train_pred_90 = y_scores >= threshold_90_precision

# %%
precision_score(y_train_5, y_train_pred_90)


# %%
recall_score(y_train_5, y_train_pred_90)

# %%
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# %%


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], "k--")
    plt.axis([0, 1, 0, 1])
    plt.xlabel("False Positive Rate (Fall-Out)", fontsize=16)
    plt.ylabel("True Positive Rate (Recall)", fontsize=16)
    plt.grid(True)


plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
plt.plot([4.837e-3, 4.837e-3], [0.0, 0.4368], "r:")
plt.plot([0.0, 4.837e-3], [0.4368, 0.4368], "r:")
plt.plot([4.837e-3], [0.4368], "ro")
plt.show()

# %%
roc_auc_score(y_train_5, y_scores)

# %%
forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(
    forest_clf, X_train, y_train_5, cv=3, method="predict_proba"
)

# %%
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# %%
plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()

# %%
roc_auc_score(y_train_5, y_scores_forest)

# %%
svm_clf = SVC()
svm_clf.fit(X_train, y_train)

# %%
svm_clf.predict([some_digit])

# %%
some_digit_scores = svm_clf.decision_function([some_digit])
some_digit_scores

# %%
np.argmax(some_digit_scores)

# %%
svm_clf.classes_

# %%
svm_clf.classes_[5]

# %%
ovr_clf = OneVsRestClassifier(SVC())
ovr_clf.fit(X_train, y_train)
ovr_clf.predict([some_digit])
len(ovr_clf.estimators)
# This one = 67
# %%
sgd_clf.fit(X_train, y_train)
sgd_clf.predict([some_digit])

# %%
sgd_clf.decision_function([some_digit])

# %%
cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")

# %%
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy")

# %%
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
conf_mx

# %%
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# %%
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# %%
np.fill_diagonal(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

# %%
y_train_large = y_train >= 7
y_train_odd = y_train % 2 == 1
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# %%
knn_clf.predict([some_digit])

# %%
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)

# %%
f1_score(y_multilabel, y_train_knn_pred, average="macro")

# %%
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

# %%
knn_clf.fit(X_train_mod, y_train_mod)
# clean_digit = knn_clf.predict([X_test_mod[some_index]])
