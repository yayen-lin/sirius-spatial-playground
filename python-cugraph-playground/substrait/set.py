"""
Patterns observed:

(p:Person)-[k:Knows]->(p2:Person)
Person ⋈ (Person ⋈ Knows)



"""

import json
import duckdb

conn = duckdb.connect(":memory:")
query = "SELECT * FROM graph_table(pg MATCH (p:Person)-[k:Knows]->(p2:Person) COLUMNS (p.name, p2.name));"

conn.execute("INSTALL substrait FROM community;")
conn.execute("INSTALL duckpgq FROM community;")
conn.execute("LOAD substrait;")
conn.execute("LOAD duckpgq;")

conn.execute("CREATE TABLE Person (id INTEGER, name VARCHAR);")
conn.execute("INSERT INTO Person VALUES (1, 'Alice'), (2, 'Bob'), (3, 'Charlie');")
conn.execute("CREATE TABLE Knows (src INTEGER, dst INTEGER);")
conn.execute("INSERT INTO Knows VALUES (1, 2), (2, 3), (1, 3);")
conn.execute("""
CREATE PROPERTY GRAPH pg
VERTEX TABLES (Person)
EDGE TABLES (Knows SOURCE KEY (src) REFERENCES Person (id)
                   DESTINATION KEY (dst) REFERENCES Person (id));
""")

result = conn.execute(query).fetchall()
for row in result:
    print(row)

substrait_result = conn.execute(f"CALL get_substrait_json('{query}');").fetchone()
substrait_json = substrait_result[0]
print(json.dumps(json.loads(substrait_json), indent=4))
