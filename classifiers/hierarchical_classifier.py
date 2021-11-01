from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT
from sklearn.metrics  import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn import svm



def hierarchical_classifier(x_train, y_train, x_test, y_test):
    class_hierarchy = {
        ROOT: ["clp","cmp"],
        "clp": ["Natural killer cell","small lymphocytes"],
        "cmp": ["Megakaryocyte", "Monocytes", "apc"],
        "small lymphocytes": ["B cell", "T cell"],
        "T cell": ["CD4+ T cell", "Cytotoxic T cell"],
        "Monocytes": ["CD14+ monocyte", "CD16+ monocyte"],
        "apc":["Dendritic cell", "pdc"], 
        "pdc": ["Plasmacytoid dendritic cell"]
        }
    base_estimator = make_pipeline(
        TruncatedSVD(n_components=65),
        svm.SVC(probability= True)
    )

    clf = HierarchicalClassifier(
        base_estimator=base_estimator,
        class_hierarchy=class_hierarchy
    )

    clf.fit(x_train, y_train)
    perf = f1_score(clf.predict(x_test), y_test, average="weighted")
    return clf, perf