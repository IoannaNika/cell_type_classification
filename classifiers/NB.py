
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import FactorAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics  import f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB


def  gnb(x_train, y_train, x_test, y_test):
    steps = [('fa', FactorAnalysis(n_components=20)), ('m', GaussianNB())]
    model = Pipeline(steps=steps)
    model.fit(x_train, y_train)

    acc = f1_score(model.predict(x_test), y_test, average= "weighted")
    return model, acc

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

