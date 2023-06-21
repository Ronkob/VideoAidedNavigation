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
        if self.cov:
            self.weight = np.linalg.det(cov)


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
        Find shortest path from c_i to c_n using Dijkstra algorithm.
        """
        dists = [float('inf') for _ in range(len(self.size))]
        dists[i] = 0
        prevs = np.zeros(self.size)
        visited = [False] * self.size

        while visited[n] is False:
            # Find the vertex with min distance to i from the set of vertices not yet visited
            min_dist, min_dist_vertex = float('inf'), None
            for v in range(self.size):
                # dist[v] = distance from c_v to c_n
                if dists[v] < min_dist and visited[v] is False:
                    min_dist, min_dist_vertex = dists[v], v
            visited[min_dist_vertex] = True

            for v in range(self.size):
                if self.adj_mat[min_dist_vertex, v] and visited[v] is False:
                    if dists[min_dist_vertex] + self.adj_mat[min_dist_vertex, v].weight < dists[v]:
                        # Update the distances based on neighbors of min_dist_vertex
                        dists[v] = dists[min_dist_vertex] + self.adj_mat[min_dist_vertex, v].cn_pose, ci_pose, rel_cov
                        prevs[v] = min_dist_vertex

        # Calculate the min path from c_n to c_i
        if not prevs[n]: return [n]

        path = []
        while i != n:
            path.append(i)
            i = prevs[i]
        path.append(n)
        path.reverse()

        return path

    def add_new_edge(self, s, t, cov):
        """
        Add a new edge to the vertex graph.
        """
        self.adj_mat[s, t] = Edge(s, t, cov)


    @staticmethod
    def calc_cov_along_path(path, rel_covs):
        """
        Sum the covariances along the path to get an estimate of the relative covariance.
        """
        rel_cov = np.zeros((6, 6))
        for j in range(len(path) - 1):
            rel_cov += rel_covs[j]
        return rel_cov
