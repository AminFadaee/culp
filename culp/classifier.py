import numpy

from culp.abstracts import ClassScoresStrategy
from culp.leg import Leg
from culp.similarity_strategies import KNNSimilarityEdgesStrategy


def culp(train, labels, test, class_scores_strategy: ClassScoresStrategy, similarity, k, n_jobs=1):
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
    similarity_edges_strategy = KNNSimilarityEdgesStrategy(train, test, similarity, k, n_jobs)
    leg = Leg(n, m, c, labels, similarity_edges_strategy)
    class_scores_strategy.load_leg(leg)
    similarities = class_scores_strategy.compute_class_scores()
    prediction = numpy.argmax(similarities, axis=1)
    return prediction
