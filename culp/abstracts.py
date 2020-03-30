from abc import abstractmethod, ABC

from culp.leg import Leg


class SimilarityEdgesStrategy(ABC):
    @abstractmethod
    def find_neighbors(self):
        pass


class ClassScoresStrategy(ABC):
    leg: Leg
    class_indices: range
    test_indices: range

    def load_leg(self, leg: Leg):
        self.leg = leg
        self.class_indices = range(self.leg.n + self.leg.m, self.leg.n + self.leg.m + self.leg.c)
        self.test_indices = range(self.leg.n, self.leg.n + self.leg.m)

    @abstractmethod
    def compute_class_scores(self):
        pass
