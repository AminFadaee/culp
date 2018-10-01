import numpy
import networkx as nx

epsilon = 10e-10


def common_neighbors(G, test_indices: tuple, class_indices: tuple):
    assert (isinstance(test_indices, tuple) and len(test_indices) == 2)
    assert (isinstance(class_indices, tuple) and len(class_indices) == 2)
    N = len(range(*test_indices))
    C = len(range(*class_indices))
    similarities = numpy.zeros((N, C), dtype=float)
    for class_index, j in enumerate(range(*class_indices)):
        for test_index, i in enumerate(range(*test_indices)):
            similarities[test_index, class_index] = len(list(nx.common_neighbors(G, i, j)))

    prediction = numpy.argmax(similarities, axis=1)
    return prediction


def adamic_adar(G, test_indices: tuple, class_indices: tuple):
    assert (isinstance(test_indices, tuple) and len(test_indices) == 2)
    assert (isinstance(class_indices, tuple) and len(class_indices) == 2)
    N = len(range(*test_indices))
    C = len(range(*class_indices))
    similarities = numpy.zeros((N, C), dtype=float)
    for class_index, j in enumerate(range(*class_indices)):
        for test_index, i in enumerate(range(*test_indices)):
            for n in nx.common_neighbors(G, i, j):
                similarities[test_index, class_index] += 1 / (numpy.log(G.degree[n]) + 10e-10)
    prediction = numpy.argmax(similarities, axis=1)
    return prediction


def resource_allocation_index(G, test_indices: tuple, class_indices: tuple):
    assert (isinstance(test_indices, tuple) and len(test_indices) == 2)
    assert (isinstance(class_indices, tuple) and len(class_indices) == 2)
    N = len(range(*test_indices))
    C = len(range(*class_indices))
    similarities = numpy.zeros((N, C), dtype=float)
    for class_index, j in enumerate(range(*class_indices)):
        for test_index, i in enumerate(range(*test_indices)):
            for n in nx.common_neighbors(G, i, j):
                similarities[test_index, class_index] += 1 / G.degree[n]

    prediction = numpy.argmax(similarities, axis=1)
    return prediction


def compatibility_score(G, test_indices: tuple, class_indices: tuple):
    assert (isinstance(test_indices, tuple) and len(test_indices) == 2)
    assert (isinstance(class_indices, tuple) and len(class_indices) == 2)
    N = len(range(*test_indices))
    C = len(range(*class_indices))
    similarities = numpy.zeros((N, C), dtype=float)
    for class_index, j in enumerate(range(*class_indices)):
        for test_index, i in enumerate(range(*test_indices)):
            for n in nx.common_neighbors(G, i, j):
                deg1 = G.degree[n] - len(list(nx.common_neighbors(G, n, j)))
                deg2 = G.degree[n] - len(list(nx.common_neighbors(G, n, i)))
                similarities[test_index, class_index] += (1 / deg1 + 1 / deg2)
    prediction = numpy.argmax(similarities, axis=1)
    return prediction
