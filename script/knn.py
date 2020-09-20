import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn import metrics
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from util import *


def knn_tuning(X_train, y_train, X_test, y_test, title):
    # create new a knn model
    knn2 = KNeighborsClassifier()

    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(1, 25)}  # ,
    # 'weights': ['uniform', 'distance'],
    # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}
    knn_gscv = GridSearchCV(knn2, param_grid, cv=5)  # fit model to data
    knn_gscv.fit(X_train, y_train)

    # check top performing n_neighbors value
    print(knn_gscv.best_params_)

    # check mean score for the top performing value of n_neighbors
    print(knn_gscv.best_score_)

    try:
        metrics.plot_roc_curve(knn_gscv, X_test, y_test)  # doctest: +SKIP
        plt.title(title)
        plt.show()
    except:
        pass

    return knn_gscv


def _heart_failure():
    data = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')

    #
    # Need to split data into train/test, and transform (normalize) values to be between [0, 1]
    #
    X = data.copy()
    X.drop(columns='DEATH_EVENT', inplace=True)

    y = data['DEATH_EVENT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, random_state=42)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_train_minmax = min_max_scaler.fit_transform(X_train)
    X_test_minmax = min_max_scaler.fit_transform(X_test)

    scores = []
    x = np.arange(1, 25)
    for i in range(1, 25, 1):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train_minmax, y_train)
        scores.append(model.score(X_test_minmax, y_test))

    plt.plot(x, scores)
    plt.xlabel('n Neighbors')
    plt.ylabel('Accuracy')
    plt.title('KNN Model Accuracy Across k Neighbors')
    plt.show()

    model = knn_tuning(X_train_minmax, y_train, X_test_minmax,
                       y_test, 'KNN Accuracy Normalized Data')

    pred = model.predict(X_test_minmax)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, pred)
    print(metrics.auc(fpr, tpr))

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    plot_learning_curve(KNeighborsClassifier(n_neighbors=6),
                        'Heart Failure Dataset Learning Curve (cv=5)',
                        X_train_minmax, y_train, axes=axes[0], ylim=(0.0, 1.1),
                        cv=5, n_jobs=4)
    plot_learning_curve(KNeighborsClassifier(n_neighbors=6),
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

    scores = []
    x = np.arange(1, 25)
    for i in range(1, 25, 1):
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train_minmax, y_train)
        scores.append(model.score(X_test_minmax, y_test))

    plt.plot(x, scores)
    plt.xlabel('n Neighbors')
    plt.ylabel('Accuracy')
    plt.title('KNN Model Accuracy Across n Neighbors')
    plt.show()

    model = knn_tuning(X_train_minmax, y_train, X_test_minmax,
                       y_test, 'KNN Accuracy Normalized Data')

    y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
    y_pred = model.predict(X_test_minmax)

    y_pred = label_binarize(y_pred, classes=[0, 1, 2, 3, 4, 5, 6])

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

    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(X_train_minmax, y_train)

    y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
    y_pred = model.predict(X_test_minmax)

    y_pred = label_binarize(y_pred, classes=[0, 1, 2, 3, 4, 5, 6])

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

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    plot_learning_curve(KNeighborsClassifier(n_neighbors=6),
                        'Heart Failure Dataset Learning Curve (cv=5)',
                        X_train_minmax, y_train, axes=axes[0], ylim=(0.0, 1.1),
                        cv=5, n_jobs=4)
    plot_learning_curve(KNeighborsClassifier(n_neighbors=6),
                        'Heart Failure Dataset Learning Curve (cv=10)',
                        X_train_minmax, y_train, axes=axes[1], ylim=(0.0, 1.1),
                        cv=10, n_jobs=4)


def knn():
    _heart_failure()
    _obesity()

