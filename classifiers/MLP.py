from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import FactorAnalysis
from sklearn.pipeline import Pipeline
from sklearn.metrics  import f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB



def  mlp(x_train, y_train, x_test, y_test):
    steps = [('fa',FactorAnalysis(n_components=65)), ('m', MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))]
    model = Pipeline(steps=steps)
    model.fit(x_train, y_train)

    acc = f1_score(model.predict(x_test), y_test, average= "weighted")
    return model, acc


