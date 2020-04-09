import numpy

from .abstracts import ClassScoresStrategy


class CommonNeighborsStrategy(ClassScoresStrategy):
    __name__ = 'CN'

    def find_score(self, class_node, test_node):
        return self.leg.count_common_neighbors(class_node, test_node)


class AdamicAdarStrategy(ClassScoresStrategy):
    __name__ = 'AA'

    def find_score(self, class_node, test_node):
        score = 0
        for n in self.leg.common_neighbors(class_node, test_node):
            score += 1 / (numpy.log(self.leg.degree(n)) + 10e-10)
        return score


class ResourceAllocationStrategy(ClassScoresStrategy):
    __name__ = 'RA'

    def find_score(self, class_node, test_node):
        score = 0
        for n in self.leg.common_neighbors(class_node, test_node):
            score += 1 / self.leg.degree(n)
        return score


class CompatibilityScoreStrategy(ClassScoresStrategy):
    __name__ = 'CS'

    def find_score(self, class_node, test_node):
        score = 0
        for neighbor_node in self.leg.common_neighbors(class_node, test_node):
            deg1 = self.leg.degree(neighbor_node) - self.leg.count_common_neighbors(neighbor_node, test_node)
            deg2 = self.leg.degree(neighbor_node) - self.leg.count_common_neighbors(neighbor_node, class_node)
            score += (1 / deg1 + 1 / deg2)
        return score
