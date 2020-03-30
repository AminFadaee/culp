import numpy

from .abstracts import ClassScoresStrategy, SimilarityEdgesStrategy
from .leg import Leg
from .similarity_strategies import KNNSimilarityEdgesStrategy, VectorSimilarityMetric


class CULP:
    leg: Leg

    def __init__(self, n, m, c, labels, similarity_edges_strategy: SimilarityEdgesStrategy):
        self.n = n
        self.m = m
        self.c = c
        self.labels = labels
        self.similarity_edges_strategy = similarity_edges_strategy

    def train(self):
        self.leg = Leg(self.n, self.m, self.c, self.labels, self.similarity_edges_strategy)

    def predict(self, class_scores_strategy: ClassScoresStrategy):
        class_scores_strategy.load_leg(self.leg)
        similarities = class_scores_strategy.compute_class_scores()
        prediction = numpy.argmax(similarities, axis=1)
        return prediction


class CULPUsingKNNFactory:
    @staticmethod
    def create(train_data, train_labels, test_data, classes, similarity: VectorSimilarityMetric, k, n_jobs=1):
        similarity_edges_strategy = KNNSimilarityEdgesStrategy(train_data, test_data, similarity, k, n_jobs)
        return CULP(
            n=len(train_data),
            m=len(test_data),
            c=len(classes),
            labels=train_labels,
            similarity_edges_strategy=similarity_edges_strategy
        )
