#!/usr/bin/env python3
"""
Test Single-Source Shortest Path (SSPP, weighted shortest path) with cuGraph
SSSP: Find shortest paths from one node to every node
"""

import cudf
import cugraph

edges = cudf.DataFrame({
    'src': [1, 2, 1],
    'dst': [2, 3, 3],
    'weight': [1.0, 2.0, 5.0]  # Alice→Bob=1, Bob→Charlie=2, Alice→Charlie=5
    # 'weight': [1.0, 4.0, 3.0]  # Alice→Bob=1, Bob→Charlie=4, Alice→Charlie=4
})

G = cugraph.Graph(directed=True)
G.from_cudf_edgelist(edges, source='src', destination='dst', edge_attr='weight')

# SSSP instead of BFS
starting_person = 1
distances = cugraph.sssp(G, source=starting_person)
print(distances) # Expected result: Alice→Charlie should be distance 3.0 (via Bob), not 5.0 (direct edge)
