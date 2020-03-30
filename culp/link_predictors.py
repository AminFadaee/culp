import numpy

from .abstracts import ClassScoresStrategy


class CommonNeighborsStrategy(ClassScoresStrategy):
    def compute_class_scores(self):
        similarities = numpy.zeros((self.leg.n, self.leg.c), dtype=float)
        for class_index, j in enumerate(self.class_indices):
            for test_index, i in enumerate(self.test_indices):
                similarities[test_index, class_index] = self.leg.count_common_neighbors(i, j)
        return similarities


class AdamicAdarStrategyStrategy(ClassScoresStrategy):
    def compute_class_scores(self):
        similarities = numpy.zeros((self.leg.n, self.leg.c), dtype=float)
        for class_index, j in enumerate(self.class_indices):
            for test_index, i in enumerate(self.test_indices):
                for n in self.leg.common_neighbors(i, j):
                    similarities[test_index, class_index] += 1 / (numpy.log(self.leg.degree(n)) + 10e-10)
        return similarities


class ResourceAllocationIndexStrategy(ClassScoresStrategy):
    def compute_class_scores(self):
        similarities = numpy.zeros((self.leg.n, self.leg.c), dtype=float)
        for class_index, j in enumerate(self.class_indices):
            for test_index, i in enumerate(self.test_indices):
                for n in self.leg.common_neighbors(i, j):
                    similarities[test_index, class_index] += 1 / self.leg.degree(n)
        return similarities


class CompatibilityScoreStrategy(ClassScoresStrategy):
    def compute_class_scores(self):
        similarities = numpy.zeros((self.leg.n, self.leg.c), dtype=float)
        for class_index, j in enumerate(self.class_indices):
            for test_index, i in enumerate(self.test_indices):
                for n in self.leg.common_neighbors(i, j):
                    deg1 = self.leg.degree(n) - self.leg.count_common_neighbors(n, j)
                    deg2 = self.leg.degree(n) - self.leg.count_common_neighbors(n, i)
                    similarities[test_index, class_index] += (1 / deg1 + 1 / deg2)
        return similarities
