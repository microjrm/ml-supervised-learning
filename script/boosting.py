import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from util import *
import time


def booster_tuning(X, y, X_test, y_test, classifier, param_grid, title):
    # create new a knn model
    clf = classifier
    clf_gscv = GridSearchCV(clf, param_grid, cv=5)  # fit model to data
    clf_gscv.fit(X, y)

    # check top performing n_neighbors value
    print(clf_gscv.best_params_)

    try:
        # check mean score for the top performing value of n_neighbors
        pred = clf_gscv.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
        print(metrics.auc(fpr, tpr))
        print(clf_gscv.best_score_)
        metrics.plot_roc_curve(clf_gscv, X_test, y_test)  # doctest: +SKIP
        plt.title(title)
        plt.show()
    except:
        pass
    return clf_gscv


def best_alpha(X_train, y_train, X_test, y_test, data_name):
    clfs, scores = [], []
    alphas = np.array([0, 0.05, 0.1, 0.15, 0.2, 0.25])
    for ccp_alpha in alphas:
        clf = GradientBoostingClassifier(ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf.score(X_train, y_train))
        scores.append(clf.score(X_test, y_test))

    plt.plot(alphas, clfs, label='train')
    plt.plot(alphas, scores, label='test')
    plt.xlabel('alpha')
    plt.ylabel('accuracy')
    plt.legend()
    plt.title(f'Effect of tuning parameter alpha on {data_name} accuracy (train/test)')


def _heart_failure():
    data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')

    #
    # Need to split data into train/test, and transform (normalize) values to be between [0, 1]
    #
    X = data.copy()
    X.drop(columns='DEATH_EVENT', inplace=True)

    y = data['DEATH_EVENT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

    best_alpha(X_train, y_train, X_test, y_test, 'heart failure')
    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {
        'max_features': [2, 3, 4, 5, 6, 7],
        'min_samples_leaf': [5],
        'min_samples_split': [12, 14, 16],
        'n_estimators': [25],
        'learning_rate': [0.1]
    }

    model = booster_tuning(X_train, y_train, X_test, y_test, GradientBoostingClassifier(),
                           param_grid, 'GradientBoosting Accuracy on Heart Failure Data')

    test = GradientBoostingClassifier(learning_rate=0.1, max_features=6, min_samples_leaf=5,
                                      min_samples_split=14, n_estimators=25)
    test.fit(X_train, y_train)
    pred = test.predict(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    print(metrics.auc(fpr, tpr))

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    model = GradientBoostingClassifier(learning_rate=0.1, max_depth=200, max_features=3,
                                       min_samples_leaf=5, min_samples_split=12, n_estimators=25,
                                       ccp_alpha=0)
    plot_learning_curve(model,
                        'Heart Failure Dataset Learning Curve (cv=5)',
                        X_train, y_train, axes=axes[0], ylim=(0.0, 1.1),
                        cv=5, n_jobs=4)

    plot_learning_curve(model,
                        'Heart Failure Dataset Learning Curve (cv=10)',
                        X_train, y_train, axes=axes[1], ylim=(0.0, 1.1),
                        cv=10, n_jobs=4)


def _obesity():
    df = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')

    # Handle Categorical features
    clfs = {c: preprocessing.LabelEncoder() for c in ['Gender', 'family_history_with_overweight',
                                                      'FAVC', 'CAEC', 'SMOKE',
                                                      'SCC', 'CALC', 'MTRANS', 'NObeyesdad']}

    for col, clf in clfs.items():
        df[col] = clfs[col].fit_transform(df[col])

    #
    # Need to split data into train/test, and transform (normalize) values to be between [0, 1]
    #

    X = df.drop(columns='NObeyesdad')
    y = df['NObeyesdad'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=0)

    best_alpha(X_train, y_train, X_test, y_test, 'obesity')

    param_grid = {
        'max_depth': [10, 100, 200],
        'max_features': [2, 3],
        'min_samples_leaf': [3, 4, 5],
        'min_samples_split': [8, 10, 12],
        'n_estimators': [10, 25, 50, 100],
        'learning_rate': [0.1, 0.5, 1]
    }

    start = time.time()
    model = booster_tuning(X_train, y_train, X_test, y_test, GradientBoostingClassifier(),
                           param_grid, 'GradientBoosting Accuracy Normalized Data')
    print(f'{time.time() - start} seconds to complete CV grid search')

    y_train_bin = preprocessing.label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6])
    y_test_bin = preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
    y_pred = model.predict(X_test)
    y_pred = preprocessing.label_binarize(y_pred, classes=[0, 1, 2, 3, 4, 5, 6])

    n_classes = 7
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))
    plt.legend()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])

    print(np.mean(list(roc_auc.values())))

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    model = GradientBoostingClassifier(learning_rate=0.5, max_depth=10, max_features=3,
                                       min_samples_leaf=3, min_samples_split=10, n_estimators=100,
                                       ccp_alpha=0)
    plot_learning_curve(model,
                        'Obesity Dataset Learning Curve (cv=5)',
                        X_train, y_train, axes=axes[0], ylim=(0.0, 1.1),
                        cv=5, n_jobs=4)

    plot_learning_curve(model,
                        'Obesity Dataset Learning Curve (cv=10)',
                        X_train, y_train, axes=axes[1], ylim=(0.0, 1.1),
                        cv=10, n_jobs=4)

    n_classes = 7
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.legend()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])

    np.mean(list(roc_auc.values()))


def boosting():
    _heart_failure()
    _obesity()

