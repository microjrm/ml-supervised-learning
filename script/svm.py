import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC
from util import *


def svm_tuning(X, y, X_test, y_test, classifier, param_grid, title):
    # create new a knn model
    clf = classifier
    clf_gscv = GridSearchCV(clf, param_grid, cv=5)  # fit model to data
    clf_gscv.fit(X, y)

    # check top performing n_neighbors value
    print(clf_gscv.best_params_)

    # check mean score for the top performing value of n_neighbors
    print(clf_gscv.best_score_)

    try:
        metrics.plot_roc_curve(clf_gscv, X_test, y_test)  # doctest: +SKIP
        plt.title(title)
        plt.show()
    except:
        pass
    return clf_gscv


def _heart_failure():
    data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')

    #
    # Need to split data into train/test, and transform (normalize) values to be between [0, 1]
    #
    X = data.copy()
    X.drop(columns='DEATH_EVENT', inplace=True)

    y = data['DEATH_EVENT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.fit_transform(X_test)

    for k in ['linear', 'poly']:
        model = SVC(kernel=k)
        model.fit(X_train_minmax, y_train)
        y_pred = model.predict(X_test_minmax)
        fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
        roc_auc = metrics.auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{k}, AUROC={np.round(roc_auc, 4)}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.title('AUROC of differnt kernel functions (heart failure)')
    plt.show()

    param_grid = {
        'C': np.linspace(1, 10, 2),
        'kernel': ['linear', 'poly'],
        'degree': [2, 3, 4]
        # May need more
    }

    results = svm_tuning(X_train_minmax, y_train, X_test_minmax, y_test, SVC(),
                         param_grid, 'SVM Accuracy Normalized Data')

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    model = SVC(kernel='linear')
    plot_learning_curve(model,
                        'Heart Failure Dataset Learning Curve (cv=5)',
                        X_train_minmax, y_train, axes=axes[0], ylim=(0.0, 1.1),
                        cv=5, n_jobs=4)

    plot_learning_curve(model,
                        'Heart Failure Dataset Learning Curve (cv=10)',
                        X_train_minmax, y_train, axes=axes[1], ylim=(0.0, 1.1),
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.fit_transform(X_test)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    for j, k in enumerate(['linear', 'poly']):
        model = SVC(kernel=k)
        model.fit(X_train_minmax, y_train)

        y_train_bin = preprocessing.label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6])
        y_test_bin = preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
        y_pred = model.predict(X_test_minmax)
        y_pred = preprocessing.label_binarize(y_pred, classes=[0, 1, 2, 3, 4, 5, 6])

        n_classes = 7
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_pred[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
            axes[j].plot(fpr[i], tpr[i],
                         label=f'AUROC of class {i}={np.round(np.mean(list(roc_auc.values())), 4)}')
        axes[j].legend()
        axes[j].set_title(f'SVM of obesity data with {k} kernel function')
        axes[j].plot([0, 1], [0, 1], 'k--')

        print(k)
        print(np.mean(list(roc_auc.values())))

    plt.show()

    param_grid = {
        'C': np.linspace(1, 10, 1),
        'kernel': ['linear', 'poly'],
        'degree': [2, 3, 4, 5],
        'gamma': ['scale', 'auto']
        # May need more
    }

    model = svm_tuning(X_train_minmax, y_train, X_test_minmax, y_test, SVC(),
                       param_grid, 'SVM Accuracy Normalized Data')

    param_grid = {
        'kernel': ['linear', 'poly']
    }
    model = svm_tuning(X_train_minmax, y_train, X_test_minmax, y_test, SVC(),
                       param_grid, 'SVM Accuracy Normalized Data')

    y_train_bin = preprocessing.label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6])
    y_test_bin = preprocessing.label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
    y_pred = model.predict(X_test_minmax)
    y_pred = preprocessing.label_binarize(y_pred, classes=[0, 1, 2, 3, 4, 5, 6])

    n_classes = 7
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i],
                 label=f'AUROC of class {i}={np.round(np.mean(list(roc_auc.values())), 4)}')
    plt.legend()
    plt.title(f'SVM of obesity data with poly kernel function')
    plt.plot([0, 1], [0, 1], 'k--')

    print(np.mean(list(roc_auc.values())))

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    model = SVC(kernel='poly')
    plot_learning_curve(model,
                        'Heart Failure Dataset Learning Curve (cv=5)',
                        X_train_minmax, y_train, axes=axes[0], ylim=(0.0, 1.1),
                        cv=5, n_jobs=4)

    plot_learning_curve(model,
                        'Heart Failure Dataset Learning Curve (cv=10)',
                        X_train_minmax, y_train, axes=axes[1], ylim=(0.0, 1.1),
                        cv=10, n_jobs=4)


def svm():
    _heart_failure()
    _obesity()

