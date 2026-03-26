"""mst.py — Minimum Spanning Tree construction for decentralised FL graphs.

Implements Prim's algorithm to extract the MST from a weighted adjacency matrix.
The MST is used (e.g., in client_decood_vmf.get_mst_agg_model) to identify the
most informative subset of edges for model parameter aggregation.
"""

import logging
import numpy as np

LOGGER = logging.getLogger(__name__)


def get_mst(adj, source=None):
    """Compute the Minimum Spanning Tree of a weighted graph using Prim's algorithm.

    Starting from ``source`` (default: node 0), greedily selects the minimum-
    weight edge crossing the cut between the visited and unvisited node sets,
    building up the MST one edge at a time.  Terminates after ``n_V - 1`` edges
    have been added.

    Args:
        adj: [N × N] numpy array representing the (symmetric) weighted adjacency
            matrix.  A zero entry means no edge exists between those nodes.
        source: Index of the starting node.  Defaults to node 0.

    Returns:
        np.ndarray: [N × N] binary adjacency matrix of the MST (1 where an MST
            edge exists, 0 otherwise).  The result is symmetric.

    Note:
        ``INF = 9999999`` is used as a sentinel for "no edge found yet"; it
        should comfortably exceed any real edge weight in the graph.
    """
    INF = 9999999
    n_V = adj.shape[0]

    selected = [0]*n_V #np.zeros(n_V)
    #keep the selected connections(MST) in an adj mat
    mst_mat = np.zeros((n_V,n_V))
    LOGGER.debug("Selected: %s", selected)

    if source is not None:
        selected[source] = True
    else:
        selected[0] = True

    LOGGER.debug("Selected: %s", selected)

    # set number of edge to 0
    no_edge = 0
    # the number of egde in minimum spanning tree will be
    # always less than(V - 1), where V is number of vertices in
    LOGGER.debug("Edge : Weight")

    while (no_edge < n_V - 1):
        minimum = INF
        x = -1
        y = -1
        for i in range(n_V):
            if selected[i]:
                for j in range(n_V):
                    if ((not selected[j]) and adj[i][j]):
                        # not in selected and there is an edge
                        if minimum > adj[i][j]:
                            minimum = adj[i][j]
                            x = i
                            y = j
        mst_mat[x][y]=1
        mst_mat[y][x]=1

        LOGGER.debug("%s-%s:%s", x, y, adj[x][y])
        selected[y] = True
        no_edge += 1
    return mst_mat