#!/usr/bin/env python3
"""
Test Simple Edge Traversal (SET) with cuGraph
"""

import cudf
import cugraph

# vertex table
persons = cudf.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})

# edge table
edges = cudf.DataFrame({
    'src': [1, 2, 1],
    'dst': [2, 3, 3]
})

G = cugraph.Graph(directed=True)
G.from_cudf_edgelist(edges, source='src', destination='dst')

# Join edges with person names
edge_results = edges.merge(persons, left_on='src', right_on='id', how='left')
edge_results = edge_results.rename(columns={'name': 'src_name'})
edge_results = edge_results.merge(persons, left_on='dst', right_on='id', how='left', suffixes=('', '_dst'))
edge_results = edge_results.rename(columns={'name': 'dst_name'})

print("=" * 50)
print("PERSONS:")
print(persons)
print("\nEDGES:")
print(edges)
print("=" * 50)
print(edge_results[['src_name', 'dst_name']])
