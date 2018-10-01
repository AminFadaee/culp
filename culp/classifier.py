import numpy
from culp import link_predictors
from culp import leg


def culp(train, labels, test, link_predictor, similarity, k, n_jobs=1):
    """
    creates leg using `k` and `similarity` and predicts the labels for the test data using `link_predictor`.

    :param train: numpy.array of the training data with n rows and d features (n x d)
    :param labels: numpy.array of training labels with n label c (0 <= c < C)
    :param test: numpy.array of the test data with m rows and d features (m x d)
    :param link_predictor: str, representing the link predictor; 'AA' for Adamic-Adar,
     'CN' for Common Neighbors, 'RA' for Resource Allocation and 'CS' for Compatibility
      Score
    :param similarity: str of the similarity function ('cosine', 'euclidean', 'manhattan')
    :param k: knn parameter
    :param n_jobs: number of parallel jobs for computing the knn
    :return: numpy.array of predicted labels for the test data
    """
    assert (isinstance(train, type(numpy.array([]))))  # numpy array
    assert (isinstance(test, type(numpy.array([]))))  # numpy array
    assert (test.shape[1] == train.shape[1])
    assert (isinstance(k, int))  # int
    classes = numpy.unique(labels)
    c = classes.shape[0]
    n, d = train.shape
    m, _ = test.shape
    G = leg.create_leg(train, labels, test, c, similarity, k, n_jobs)
    predictor = {
        'AA': link_predictors.adamic_adar,
        'CN': link_predictors.common_neighbors,
        'RA': link_predictors.resource_allocation_index,
        'CS': link_predictors.compatibility_score,
    }[link_predictor]
    test_indices, class_indices = (n, n + m), (n + m, n + m + c)
    prediction = predictor(G, test_indices, class_indices)
    return prediction
