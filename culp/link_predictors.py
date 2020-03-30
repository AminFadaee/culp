import numpy


def common_neighbors(leg, test_indices: tuple, class_indices: tuple):
    assert (isinstance(test_indices, tuple) and len(test_indices) == 2)
    assert (isinstance(class_indices, tuple) and len(class_indices) == 2)
    N = len(range(*test_indices))
    C = len(range(*class_indices))
    similarities = numpy.zeros((N, C), dtype=float)
    for class_index, j in enumerate(range(*class_indices)):
        for test_index, i in enumerate(range(*test_indices)):
            similarities[test_index, class_index] = leg.count_common_neighbors(i, j)

    prediction = numpy.argmax(similarities, axis=1)
    return prediction


def adamic_adar(leg, test_indices: tuple, class_indices: tuple):
    assert (isinstance(test_indices, tuple) and len(test_indices) == 2)
    assert (isinstance(class_indices, tuple) and len(class_indices) == 2)
    N = len(range(*test_indices))
    C = len(range(*class_indices))
    similarities = numpy.zeros((N, C), dtype=float)
    for class_index, j in enumerate(range(*class_indices)):
        for test_index, i in enumerate(range(*test_indices)):
            for n in leg.common_neighbors(i, j):
                similarities[test_index, class_index] += 1 / (numpy.log(leg.degree(n)) + 10e-10)
    prediction = numpy.argmax(similarities, axis=1)
    return prediction


def resource_allocation_index(leg, test_indices: tuple, class_indices: tuple):
    assert (isinstance(test_indices, tuple) and len(test_indices) == 2)
    assert (isinstance(class_indices, tuple) and len(class_indices) == 2)
    N = len(range(*test_indices))
    C = len(range(*class_indices))
    similarities = numpy.zeros((N, C), dtype=float)
    for class_index, j in enumerate(range(*class_indices)):
        for test_index, i in enumerate(range(*test_indices)):
            for n in leg.common_neighbors(i, j):
                similarities[test_index, class_index] += 1 / leg.degree(n)

    prediction = numpy.argmax(similarities, axis=1)
    return prediction


def compatibility_score(leg, test_indices: tuple, class_indices: tuple):
    assert (isinstance(test_indices, tuple) and len(test_indices) == 2)
    assert (isinstance(class_indices, tuple) and len(class_indices) == 2)
    N = len(range(*test_indices))
    C = len(range(*class_indices))
    similarities = numpy.zeros((N, C), dtype=float)
    for class_index, j in enumerate(range(*class_indices)):
        for test_index, i in enumerate(range(*test_indices)):
            for n in leg.common_neighbors(i, j):
                deg1 = leg.degree(n) - leg.count_common_neighbors(n, j)
                deg2 = leg.degree(n) - leg.count_common_neighbors(n, i)
                similarities[test_index, class_index] += (1 / deg1 + 1 / deg2)
    prediction = numpy.argmax(similarities, axis=1)
    return prediction
