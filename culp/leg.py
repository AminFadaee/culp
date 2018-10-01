import numpy
import networkx as nx
from sklearn.neighbors import NearestNeighbors


def create_leg(train, labels, test, c, similarity, k, n_jobs=1):
    """
    Computes the similarity between each points in the data and creates an undirected edge if it's in the ``k`` nearest
    neighbor of a node. The last ``c`` nodes of the output LEG correspond to the classes.

    :param train: numpy array of training data
    :param labels: label of the data
    :param test: numpy array of test data
    :param c: number of classes
    :param similarity: str of the similarity function ('cosine', 'euclidean', 'manhattan')
    :param k: knn parameter
    :param n_jobs: number of parallel jobs for computing the knn
    :return: LEG G
    """
    G = nx.Graph()
    data = numpy.concatenate((train, test), axis=0)
    n, d = train.shape
    m, _ = test.shape
    for i in range(n + m + c):
        G.add_node(i)
    nn = NearestNeighbors(n_neighbors=k + 1, metric=similarity, n_jobs=n_jobs)
    nn.fit(data)
    bests = nn.kneighbors(data)[1][:, 1:]
    for i in range(n + m):
        if i < n:
            G.add_edge(i, n + m + labels[i])
        best_k_indices = bests[i]
        for index in best_k_indices:
            G.add_edge(i, index)
    return G
