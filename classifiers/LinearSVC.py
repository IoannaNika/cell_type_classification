from sklearn import neighbors
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics  import f1_score
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import TruncatedSVD
from sklearn import svm




def lsvc(x_train, y_train, x_test, y_test):
    
    # define model
    steps = [('fa', FactorAnalysis(n_components=65)), ('m', svm.LinearSVC())]
    model = Pipeline(steps=steps, verbose=1)

    model.fit(x_train, y_train)

    acc = f1_score(model.predict(x_test), y_test, average= "weighted")
    return model, acc
