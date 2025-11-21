#!/usr/bin/env python3
"""
Test Breadth First Search (BFS, unweighted shortest path) with cuGraph
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

starting_person = 1
G = cugraph.Graph(directed=True)
G.from_cudf_edgelist(edges, source='src', destination='dst', renumber=True) # , renumber=False

# BFS returns distances from source
distances = cugraph.bfs(G, start=starting_person)
# Merge with person names for readable output
distances_with_names = distances.merge(persons, left_on='vertex', right_on='id', how='left')

print("=" * 50)
print("PERSONS:")
print(persons)
print("\nEDGES:")
print(edges)
print("=" * 50)
starting_name = persons.loc[persons['id'] == starting_person, 'name'].iloc[0]
print(f"DISTANCE FROM {starting_name}:")
print(distances)
