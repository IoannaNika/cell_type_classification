from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics  import f1_score
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import TruncatedSVD


def knn(x_train, y_train, x_test, y_test):
    n = 16
    # define model
    steps = [('trsvd', TruncatedSVD(n_components=11)), ('m', neighbors.KNeighborsClassifier(n, n_jobs=-1, metric="manhattan"))]
    model = Pipeline(steps=steps, verbose=1)
    # apply knn
    model.fit(x_train, y_train)
    # Accuracy
    acc = f1_score(model.predict(x_test), y_test, average= "weighted")
    return model, acc
