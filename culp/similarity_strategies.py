from enum import Enum

import numpy
from sklearn.neighbors import NearestNeighbors

from .abstracts import SimilarityEdgesStrategy


class VectorSimilarityMetric(Enum):
    cosine = 'cosine'
    manhattan = 'manhattan'
    euclidean = 'euclidean'


class KNNSimilarityEdgesStrategy(SimilarityEdgesStrategy):
    def __init__(self, train, test, similarity: VectorSimilarityMetric, k: int, n_jobs=1):
        self.train = train
        self.test = test
        self.similarity = similarity
        self.k = k
        self.n_jobs = n_jobs

    def find_neighbors(self):
        data = numpy.concatenate((self.train, self.test), axis=0)
        nn = NearestNeighbors(n_neighbors=self.k + 1, metric=self.similarity.value, n_jobs=self.n_jobs)
        nn.fit(data)
        neighbors = nn.kneighbors(data)[1][:, 1:]
        return neighbors
