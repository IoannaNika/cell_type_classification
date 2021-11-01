from sklearn.pipeline import Pipeline
from sklearn.metrics  import f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier





def  sdg(x_train, y_train, x_test, y_test):
    steps = [('trsvd',TruncatedSVD(n_components=150)), ('m', SGDClassifier(fit_intercept = False, class_weight = "balanced", average = True ))]
    model = Pipeline(steps=steps)
    model.fit(x_train, y_train)

    acc = f1_score(model.predict(x_test), y_test, average= "weighted")
    return model, acc

