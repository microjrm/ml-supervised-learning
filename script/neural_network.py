from util import *
import numpy as np
np.random.seed(1337)
import pandas as pd
import keras
from keras.layers import Dense
from keras.optimizers import SGD
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.neural_network import MLPClassifier


def nn_tuning(X_train, y_train, X_test, y_test, title):
    # create new a knn model
    nn = MLPClassifier()

    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {'hidden_layer_sizes': [(16, 2), (16, 16, 16, 16, 2),
                                         (32, 2), (32, 32, 32, 32, 2),
                                         (128, 2), (128, 128, 128, 128, 2),
                                         (512, 2), (512, 512, 32, 32, 2)],  # ,
                  'activation': ['logistic', 'relu'],
                  'solver': ['sgd', 'adam'],
                  'max_iter': [10000]}
    nn_gscv = GridSearchCV(nn, param_grid, cv=5)  # fit model to data
    nn_gscv.fit(X_train, y_train)

    # check top performing n_neighbors value
    print(nn_gscv.best_params_)

    # check mean score for the top performing value of n_neighbors
    print(nn_gscv.best_score_)

    try:
        metrics.plot_roc_curve(nn_gscv, X_test, y_test)  # doctest: +SKIP
        plt.title(title)
        plt.show()
    except:
        pass

    return nn_gscv


def _heart_failure():
    df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
    X = df.drop(columns='DEATH_EVENT')
    y = df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler.transform(X_test)

    clf = MLPClassifier(hidden_layer_sizes=(32, 2), random_state=0,
                        max_iter=10000, warm_start=True, activation='relu', solver='adam')
    clf.fit(X_train, y_train)
    scores = clf.predict(X_test)
    print(scores)
    metrics.plot_roc_curve(clf, X_test, y_test)

    model = nn_tuning(X_train, y_train, X_test,
                      y_test, 'Neural Network Accuracy Normalized Data')

    metrics.plot_roc_curve(model, X_test, y_test)

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    plot_learning_curve(MLPClassifier(hidden_layer_sizes=(128, 2), random_state=0,
                                      max_iter=10000, warm_start=True, activation='relu', solver='sgd'),
                        'Heart Failure Dataset Learning Curve (cv=5)',
                        X_train, y_train, axes=axes[0], ylim=(0.0, 1.1),
                        cv=5, n_jobs=4)
    plot_learning_curve(MLPClassifier(hidden_layer_sizes=(128, 2), random_state=0,
                                      max_iter=10000, warm_start=True, activation='relu', solver='sgd'),
                        'Heart Failure Dataset Learning Curve (cv=10)',
                        X_train, y_train, axes=axes[1], ylim=(0.0, 1.1),
                        cv=10, n_jobs=4)

    input_shape = len(X.columns)
    model = keras.Sequential()
    model.add(Dense(input_shape, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='binary_crossentropy', optimizer='Adam')
    history = model.fit(X_train, y_train.values,
                        validation_data=(X_test, y_test.values), epochs=10, verbose=0)

    x = np.arange(10)
    plt.xlabel('epoch')
    plt.title('Neural Network Train/Validate Loss Function of Heart Failure Data')
    plt.plot(x, history.history['loss'], label='loss')
    plt.plot(x, history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.show()


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

    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)

    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test = scaler.transform(X_test)

    model = nn_tuning(X_train, y_train, X_test,
                      y_test, 'Neural Network Accuracy Normalized Data')

    model = MLPClassifier(hidden_layer_sizes=(16, 2), random_state=0,
                          max_iter=10000, warm_start=True, activation='relu', solver='sgd')
    model.fit(X_train, y_train)

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
    plt.title('AUROC of Neural Network on Obesity Test Data')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    np.mean(list(roc_auc.values()))

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    plot_learning_curve(MLPClassifier(hidden_layer_sizes=(16, 2), random_state=0,
                                      max_iter=10000, warm_start=True, activation='relu', solver='sgd'),
                        'Obesity Dataset Learning Curve (cv=5)',
                        X_train, y_train, axes=axes[0], ylim=(0.0, 1.1),
                        cv=5, n_jobs=4)
    plot_learning_curve(MLPClassifier(hidden_layer_sizes=(16, 2), random_state=0,
                                      max_iter=10000, warm_start=True, activation='relu', solver='sgd'),
                        'Obesity Dataset Learning Curve (cv=10)',
                        X_train, y_train, axes=axes[1], ylim=(0.0, 1.1),
                        cv=10, n_jobs=4)

    input_shape = len(X.columns)
    model = keras.Sequential()
    model.add(Dense(input_shape, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(7, activation='relu'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam')
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test), epochs=100, verbose=0)

    x = np.arange(100)
    plt.xlabel('epoch')
    plt.title('Neural Network Train/Validate Loss Function of Obesity Data')
    plt.plot(x, history.history['loss'], label='loss')
    plt.plot(x, history.history['val_loss'], label='validation loss')
    plt.legend()
    plt.show()


def neural_net():
    _heart_failure()
    _obesity()
