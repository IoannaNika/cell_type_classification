from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics  import f1_score
from sklearn.decomposition import TruncatedSVD

def logreg(x_train, y_train, x_test, y_test):
    steps = [('trsvd', TruncatedSVD(n_components=80)), ('m', LogisticRegression(n_jobs=-1))]
    model = Pipeline(steps=steps)
    model.fit(x_train, y_train)

    #Print accuracy
    acc = f1_score(model.predict(x_test), y_test, average= "weighted")
    return model, acc