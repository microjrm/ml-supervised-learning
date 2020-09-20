from script.neural_network import neural_net
from script.boosting import boosting
from script.svm import svm
from script.decision_tree import decision_tree
from script.knn import knn


def main():
    algs = [decision_tree(),
            knn(),
            neural_net(),
            svm(),
            boosting()]

    for alg in algs:
        try:
            alg
        except RuntimeError as e:
            print(e)


if __name__ == '__main__':
    main()
