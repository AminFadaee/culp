from abc import abstractmethod, ABC


class SimilarityEdgesStrategy(ABC):
    @abstractmethod
    def find_neighbors(self):
        pass
