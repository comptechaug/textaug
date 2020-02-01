from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy, seaborn, pandas
from matplotlib import pyplot

class Metrics:
    
    def bootstrap_toy( y_pred, y_test, num=1000):
        answers = (y_pred == y_test.values)
        samples_num = len(answers)
        means = numpy.array([numpy.mean( answers[ numpy.random.randint(len(answers), size=samples_num) ] ) for _ in range(num)])
        result = (f'Accuracy mean: {means.mean():.3f} +/- {means.std()*2:.3f} (95% conf.)')
        return result
        
    def confusion_matrix( y_pred, y_test, labels=True ):
        conf_mat = confusion_matrix(y_test, y_pred)
        seaborn.set(rc={'figure.figsize':(9,9)}, font_scale=2)
        g = seaborn.heatmap(conf_mat, xticklabels=True, yticklabels=True, annot=True)
        g.set(xlabel='Predicted label', ylabel='True label')
        return conf_mat

    #Learning curve: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

class Classifiers:
    
    def log_reg(X_train, X_test, y_train, y_test):
        clf_aug = LogisticRegression(max_iter=400, verbose=1, n_jobs=-1)
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        # print( 'Log reg.', Metrics.bootstrap_toy(y_pred, y_test, 100) )
        return accuracy_score(y_test, y_pred)
        
    def random_forest(X_train, X_test, y_train, y_test):
        clf_aug = RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=1)
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        # print( 'Random forest.', Metrics.bootstrap_toy(y_pred, y_test, 100) )
        return accuracy_score(y_test, y_pred)
    
    #SVC
        
    def svc(X_train, X_test, y_train, y_test):
        clf_aug = SVC(verbose=1)
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        # print( 'SVC.', Metrics.bootstrap_toy(y_pred, y_test, 100) )
        return accuracy_score(y_test, y_pred)
        
    #PCA
    
    def pca_log_reg(X_train, X_test, y_train, y_test):
        split_index = X_train.shape[0]
        if isinstance(X_train, pandas.core.frame.DataFrame):
            X = numpy.vstack((X_train.values, X_test.values))
        else:
            X = numpy.vstack((X_train.toarray(), X_test.toarray()))
        pca = PCA(n_components=min(X.shape[0], 200))
        X = pca.fit_transform(X)
        return Classifiers.log_reg(X[:split_index,], X[split_index:,], y_train, y_test)
    
    
    #Naive Bayes
    #MLPClassifier
    #KNeighborsClassifier
        
        
    