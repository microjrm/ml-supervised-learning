from util import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree, metrics
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, label_binarize


def dtree_grid_search(X, y, X_test, y_test, alpha, nfolds):
    param_grid = {'criterion':['gini','entropy'],
                  'max_depth': np.arange(3, 10)}
    dtree_model=tree.DecisionTreeClassifier(ccp_alpha=alpha)
    dtree_gscv = GridSearchCV(dtree_model, param_grid, cv=nfolds)
    dtree_gscv.fit(X, y)
    print(dtree_gscv.score(X_test, y_test))
    return dtree_gscv.best_params_, dtree_gscv.predict(X_test)


def _heart_failure():
    df = pd.read_csv('data/heart_failure_clinical_records_dataset.csv')
    X = df.drop(columns='DEATH_EVENT')
    y = df['DEATH_EVENT']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        train_size=0.75,
                                                        random_state=0)
    clf = tree.DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities

    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")

    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]))

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()

    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))
    plot_learning_curve(tree.DecisionTreeClassifier(ccp_alpha=0.012, max_depth=7),
                        'Heart Failure Dataset Learning Curve (cv=5)',
                        X_train, y_train, axes=axes[0], ylim=(0.0, 1.1),
                        cv=5, n_jobs=4)
    plot_learning_curve(tree.DecisionTreeClassifier(ccp_alpha=0.012, max_depth=7),
                        'Heart Failure Dataset Learning Curve (cv=5)',
                        X_train, y_train, axes=axes[1], ylim=(0.0, 1.1),
                        cv=10, n_jobs=4)
    t, x = dtree_grid_search(X_train, y_train, X_test, y_test, 0.012, 5)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, x)
    metrics.auc(fpr, tpr)
    metrics.plot_roc_curve(clf, X_test, y_test)
    print(clf.score(X_test, y_test))


def _obesity():
    df = pd.read_csv('data/ObesityDataSet_raw_and_data_sinthetic.csv')
    clfs = {c: LabelEncoder() for c in ['Gender', 'family_history_with_overweight',
                                        'FAVC', 'CAEC', 'SMOKE',
                                        'SCC', 'CALC', 'MTRANS', 'NObeyesdad']}

    for col, clf in clfs.items():
        df[col] = clfs[col].fit_transform(df[col])

    X = df.drop(columns='NObeyesdad')
    y = df['NObeyesdad'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75,
                                                        random_state=0)

    clf = tree.DecisionTreeClassifier(random_state=0)
    path = clf.cost_complexity_pruning_path(X_train, y_train)
    ccp_alphas, impurities = path.ccp_alphas, path.impurities
    fig, ax = plt.subplots()
    ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
    ax.set_xlabel("effective alpha")
    ax.set_ylabel("total impurity of leaves")
    ax.set_title("Total Impurity vs effective alpha for training set")
    clfs = []
    for ccp_alpha in ccp_alphas:
        clf = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
        clf.fit(X_train, y_train)
        clfs.append(clf)
    print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]))

    clfs = clfs[:-1]
    ccp_alphas = ccp_alphas[:-1]

    node_counts = [clf.tree_.node_count for clf in clfs]
    depth = [clf.tree_.max_depth for clf in clfs]
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
    ax[0].set_xlabel("alpha")
    ax[0].set_ylabel("number of nodes")
    ax[0].set_title("Number of nodes vs alpha")
    ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
    ax[1].set_xlabel("alpha")
    ax[1].set_ylabel("depth of tree")
    ax[1].set_title("Depth vs alpha")
    fig.tight_layout()
    train_scores = [clf.score(X_train, y_train) for clf in clfs]
    test_scores = [clf.score(X_test, y_test) for clf in clfs]

    fig, ax = plt.subplots()
    ax.set_xlabel("alpha")
    ax.set_ylabel("accuracy")
    ax.set_title("Accuracy vs alpha for training and testing sets")
    ax.plot(ccp_alphas, train_scores, marker='o', label="train",
            drawstyle="steps-post")
    ax.plot(ccp_alphas, test_scores, marker='o', label="test",
            drawstyle="steps-post")
    ax.legend()
    plt.show()

    t, x = dtree_grid_search(X_train, y_train, X_test, y_test, 0.001, 5)
    y_train_bin = label_binarize(y_train, classes=[0, 1, 2, 3, 4, 5, 6])
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4, 5, 6])
    x = label_binarize(x, classes=[0, 1, 2, 3, 4, 5, 6])
    n_classes = 7
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_test_bin[:, i], x[:, i])
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
    plot_learning_curve(tree.DecisionTreeClassifier(ccp_alpha=0.001, max_depth=9),
                        'Obesity Dataset Learning Curve (cv=5)',
                        X_train, y_train, axes=axes[0], ylim=(0.0, 1.1),
                        cv=5, n_jobs=4)
    plot_learning_curve(tree.DecisionTreeClassifier(ccp_alpha=0.001, max_depth=9),
                        'Obesity Dataset Learning Curve (cv=5)',
                        X_train, y_train, axes=axes[1], ylim=(0.0, 1.1),
                        cv=10, n_jobs=4)


def decision_tree():
    _heart_failure()
    _obesity()








