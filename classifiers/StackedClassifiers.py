from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import neighbors
from sklearn.metrics  import f1_score



def stackedClassifier(x_train, y_train, x_test, y_test):

    dt = DecisionTreeClassifier(criterion = "entropy")
    linearsvc = svm.LinearSVC()
    sdg = SGDClassifier(fit_intercept = False, class_weight = "balanced", average = True )
    knn = neighbors.KNeighborsClassifier(16, n_jobs=-1)

    estimator_list = [
        ('dt', dt),
        ('linearsvc', linearsvc),
        ('sdg', sdg),
        ('knn', knn)
    ]

    stack_model = StackingClassifier(
        estimators=estimator_list, final_estimator=LogisticRegression()
    )

    stack_model.fit(x_train, y_train)

    y_test_pred = stack_model.predict(x_test)

    stack_model_test_f1 = f1_score(y_test, y_test_pred, average='weighted') # Calculate F1-score

    return stack_model_test_f1
