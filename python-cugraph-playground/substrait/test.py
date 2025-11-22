"""
WITH RECURSIVE paths (startNode, endNode, path) AS (
    SELECT Person1Id AS startNode, Person2Id AS endNode, ARRAY[Person1Id, Person2Id] AS path
        FROM Person_knows_Person JOIN Person p1 ON p1.id = Person_knows_Person.Person1Id WHERE p1.firstName = 'Hossein'
    UNION ALL (
        WITH paths AS (TABLE paths)
            SELECT paths.startNode AS startNode, Person2Id As endNode, array_append(path, Person2Id) AS path
            FROM paths JOIN Person_knows_Person ON paths.endNode = Person_knows_Person.Person1Id
            WHERE NOT EXISTS (SELECT true FROM paths previous_paths
                                JOIN person p2 ON p2.id = Person_knows_Person.Person2Id
                                WHERE p2.firstName = 'Hossein' OR Person_knows_Person.Person2Id = previous_paths.endNode)))
SELECT count (p2.id) AS cp2
FROM Person p1
JOIN paths     ON paths.startNode = p1.id
JOIN person p2 ON p2.id = paths.endNode;
-- JOIN livesIn l ON p2.id = l.personid
-- JOIN city c    ON c.id = l.id AND c.name = 'Utrecht';
"""

import json
import duckdb

conn = duckdb.connect(':memory:')
query = """
        WITH RECURSIVE paths (startNode, endNode, path) AS (
            SELECT Person1Id AS startNode, Person2Id AS endNode, ARRAY[Person1Id, Person2Id] AS path
        FROM Person_knows_Person JOIN Person p1 ON p1.id = Person_knows_Person.Person1Id WHERE p1.firstName = 'Hossein'
        UNION ALL (
        WITH paths AS (TABLE paths)
        SELECT paths.startNode AS startNode, Person2Id As endNode, array_append(path, Person2Id) AS path
        FROM paths JOIN Person_knows_Person ON paths.endNode = Person_knows_Person.Person1Id
        WHERE NOT EXISTS (SELECT true FROM paths previous_paths
            JOIN person p2 ON p2.id = Person_knows_Person.Person2Id
            WHERE p2.firstName = 'Hossein' OR Person_knows_Person.Person2Id = previous_paths.endNode)))
        SELECT count (p2.id) AS cp2
        FROM Person p1
                 JOIN paths     ON paths.startNode = p1.id
                 JOIN person p2 ON p2.id = paths.endNode;
"""

# load substrait
conn.execute("INSTALL substrait FROM community;")
conn.execute("LOAD substrait;")

# use snb
conn.execute("ATTACH 'https://github.com/Dtenwolde/duckpgq-docs/raw/refs/heads/main/datasets/snb.duckdb';")
conn.execute("use snb;")

result = conn.execute(query).fetchall()
for row in result:
    print(row)

# Get Substrait plan
# result = conn.execute(f"CALL get_substrait_json('{query}')").fetchone()
# substrait_json = result[0]
# print(json.dumps(json.loads(substrait_json), indent=4))
