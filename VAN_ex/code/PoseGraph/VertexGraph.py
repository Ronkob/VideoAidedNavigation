import gtsam
import numpy as np

class Edge:
    """
    This class is responsible for the vertex in the vertex graph.
    """
    def __init__(self, s, t, cov=None):
        self.s = s
        self.t = t
        self.cov = cov


class VertexGraph:
    """
    This class is responsible for the vertex graph - Co probability graph.
    """
    def __init__(self, size, rel_covs):
        self.size = size
        self.adj_mat = np.array((size, size))
        self.init_vertex_graph(rel_covs)

    def init_vertex_graph(self, rel_covs):
        """
        Create the vertex graph.
        """
        for i in range(self.size):
            for j in range(self.size):
                if j == i+1:  # Adjacent vertices
                    self.adj_mat[i, j] = Edge(i, j, rel_covs[i])
                else:
                    self.adj_mat[i, j] = Edge(i, j)

    def find_shortest_path(self, n, i):
        """
        Find shortest path from c_n to c_i using Dijkstra algorithm.
        """
        dists = [float('inf')] * self.size-1
        dists.append(0)
        prevs = [float('-inf')] * self.size-1
        finished_v = [False] * self.size

        while finished_v[i] is False:
            min_dist = float('inf')
            min_dist_vertex = -1
            for v in range(self.size):
                if dists[v] < min_dist and finished_v[v] is False:
                    min_dist = dists[v]
                    min_dist_vertex = v
            finished_v[min_dist_vertex] = True

            for v in range(self.size):
                if self.adj_mat[min_dist_vertex, v] is not None:
                    if dists[min_dist_vertex] + self.adj_mat[min_dist_vertex, v].cov < dists[v]:
                        dists[v] = dists[min_dist_vertex] + self.adj_mat[min_dist_vertex, v].cov
                        prevs[v] = min_dist_vertex

        # Calculate the min path from c_n to c_i
        path = []
        while i != n:
            path.append(i)
            i = prevs[i]
        path.append(n)
        path.reverse()

        return path
