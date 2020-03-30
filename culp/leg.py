import networkx as nx

from culp.abstracts import SimilarityEdgesStrategy


class Leg:
    def __init__(self, n, m, c, labels, similarity_edges_strategy: SimilarityEdgesStrategy):
        self.n = n
        self.m = m
        self.c = c
        self.labels = labels
        self.similarity_edges_strategy = similarity_edges_strategy
        self._create()

    def _create(self):
        class_edges = self._find_class_edges()
        similarity_edges = self._find_similarity_edges()
        self._create_network_x_graph(class_edges, similarity_edges)

    def _find_class_edges(self):
        class_edges = [
            (i, self.n + self.m + self.labels[i])
            for i in range(self.n)
        ]
        return class_edges

    def _find_similarity_edges(self):
        neighbors = self.similarity_edges_strategy.find_neighbors()
        similarity_edges = [
            (i, index)
            for i in range(self.n + self.m)
            for index in neighbors[i]
        ]
        return similarity_edges

    def _create_network_x_graph(self, class_edges, similarity_edges):
        self._graph: nx.Graph = nx.Graph()
        self._graph.add_nodes_from(range(self.n + self.m + self.c))
        self._graph.add_edges_from(class_edges)
        self._graph.add_edges_from(similarity_edges)

    def common_neighbors(self, u: int, v: int):
        return list(nx.common_neighbors(self._graph, u, v))

    def count_common_neighbors(self, u: int, v: int):
        return len(self.common_neighbors(u, v))

    def degree(self, v: int):
        return self._graph.degree[v]
