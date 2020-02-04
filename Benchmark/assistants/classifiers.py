from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, confusion_matrix, accuracy_score, f1_score, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy, seaborn, pandas
from matplotlib import pyplot
from scipy.sparse import vstack
from xgboost import XGBClassifier 

class Metrics:
    
    def bootstrap_toy( y_pred, y_test, num=1000):
        answers = (y_pred == y_test.values)
        samples_num = len(answers)
        means = numpy.array([numpy.mean( answers[ numpy.random.randint(len(answers), size=samples_num) ] ) for _ in range(num)])
        result = (f'Accuracy mean: {means.mean():.3f} +/- {means.std()*2:.3f} (95% conf.)')
        return result

    def confusion_matrix( y_pred, y_test, labels=True ):
        conf_mat = confusion_matrix(y_test, y_pred) if y_test.nunique() < 30 else [[0]]
        seaborn.set(rc={'figure.figsize':(9,9)}, font_scale=2)
        g = seaborn.heatmap(conf_mat, xticklabels=True, yticklabels=True, annot=True)
        g.set(xlabel='Predicted label', ylabel='True label')
        return conf_mat

    #Learning curve: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py

class Classifiers:

    def fit_and_result(y_pred, y_test):        
        accuracy = accuracy_score(y_test, y_pred)
        balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        conf_mat = confusion_matrix(y_pred, y_test)  if y_test.nunique() < 30 else [[1]]
        return (accuracy, balanced_accuracy, f1, conf_mat)
    
    #Logistic regression
    def log_reg(X_train, X_test, y_train, y_test):
        clf_aug = LogisticRegression(max_iter=400, verbose=1, n_jobs=-1)
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        del clf_aug

        return Classifiers.fit_and_result(y_pred, y_test)
        
    #Random forest
    def random_forest(X_train, X_test, y_train, y_test):
        clf_aug = RandomForestClassifier(n_estimators=200, n_jobs=-1, verbose=1)
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        del clf_aug

        return Classifiers.fit_and_result(y_pred, y_test)
    
    #SVC
    def svc(X_train, X_test, y_train, y_test):
        clf_aug = SVC(verbose=1)
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        del clf_aug

        return Classifiers.fit_and_result(y_pred, y_test)

    #Naive Bayes
    def naive_bayes(X_train, X_test, y_train, y_test):
        clf_aug = MultinomialNB()
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        del clf_aug

        return Classifiers.fit_and_result(y_pred, y_test)

    #MLPClassifier
    def perceptron(X_train, X_test, y_train, y_test):
        print('20')
        clf_aug = MLPClassifier( hidden_layer_sizes=(20, ), verbose=1, early_stopping=True, n_iter_no_change=2, \
                                            batch_size=min(100000, X_train.shape[0]), learning_rate='invscaling', max_iter=15 )
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        del clf_aug

        return Classifiers.fit_and_result(y_pred, y_test)
    
    #Grad boosting
    def gradient_boost(X_train, X_test, y_train, y_test):
        clf_aug = GradientBoostingClassifier(n_estimators=300, n_iter_no_change=2, verbose=1)
        clf_aug.fit(X_train, y_train)
        y_pred = clf_aug.predict(X_test)
        del clf_aug

        return Classifiers.fit_and_result(y_pred, y_test)

    #XGboost
    def xgboost(X_train, X_test, y_train, y_test):
        clf_aug = XGBClassifier(n_estimators=1500, n_jobs=-1, verbose=3, colsample_bytree = 0.3)
        clf_aug.fit(X_train, y_train, early_stopping_rounds=10, eval_set=[(X_test, y_test)] )
        y_pred = clf_aug.predict(X_test)
        del clf_aug

        return Classifiers.fit_and_result(y_pred, y_test)
    