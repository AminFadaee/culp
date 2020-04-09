from abc import abstractmethod, ABC
from multiprocessing import Pool
from typing import Union

import numpy

from .leg import Leg


class SimilarityEdgesStrategy(ABC):
    @abstractmethod
    def find_neighbors(self):
        pass


class ClassScoresStrategy(ABC):
    leg: Leg
    class_indices: range
    test_indices: range
    n_jobs: int

    def load_leg(self, leg: Leg, n_jobs=1):
        self.leg = leg
        self.n_jobs = n_jobs
        self.class_indices = range(self.leg.n + self.leg.m, self.leg.n + self.leg.m + self.leg.c)
        self.test_indices = range(self.leg.n, self.leg.n + self.leg.m)

    def compute_class_scores(self):
        parameters = [
            (i, j)
            for j in self.test_indices
            for i in self.class_indices
        ]
        pool = Pool(self.n_jobs)
        results = pool.starmap(self.find_score, parameters)
        pool.close()
        pool.join()
        similarities = numpy.reshape(numpy.array(results), (self.leg.m, self.leg.c))
        return similarities

    @abstractmethod
    def find_score(self, class_node: int, test_node: int) -> Union[int, float]:
        pass
