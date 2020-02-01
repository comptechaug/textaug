from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import numpy
import seaborn

class Metrics:
    
    def bootstrap_toy( y_pred, y_test, num=1000):
        answers = (y_pred == y_test.values)
        samples_num = len(answers)
        means = numpy.array([numpy.mean( answers[ numpy.random.randint(len(answers), size=samples_num) ] ) for _ in range(num)])
        result = (f'Mean: {means.mean():.3f} +/- {means.std()*2:.3f} (95% conf.)')
        return result
        
    def confusion_matrix( y_pred, y_test, labels=True ):
        conf_mat = confusion_matrix(y_test, y_pred)
        seaborn.set(rc={'figure.figsize':(9,9)}, font_scale=2)
        seaborn.heatmap(conf_mat, xticklabels=True, yticklabels=True, annot=True)
        return conf_mat

    #Learning curve: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

class Classifiers:
    
    def log_reg(X_train, X_test, y_train, y_test):
        clf_aug = LogisticRegression(max_iter=400)
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        print( Metrics.bootstrap_toy(y_pred, y_test, 100) )
        return (y_pred, y_test, clf_aug)
        
    def random_forest(X_train, X_test, y_train, y_test):
        clf_aug = RandomForestClassifier(n_estimators=10000, max_depth = 100, n_jobs=-1, verbose=1)
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        print( Metrics.bootstrap_toy(y_pred, y_test, 100) )
        return (y_pred, y_test, clf_aug)
        
    #SVC
    #Naive Bayes
    #MLPClassifier
    #KNeighborsClassifier
        
        
    