import duckdb

conn = duckdb.connect(':memory:')

# load substrait
conn.execute("INSTALL substrait FROM community;")
conn.execute("LOAD substrait;")

# load duckpgq
conn.execute("INSTALL duckpgq FROM community;")
conn.execute("LOAD duckpgq;")

# use snb
conn.execute("ATTACH 'https://github.com/Dtenwolde/duckpgq-docs/raw/refs/heads/main/datasets/snb.duckdb';")
conn.execute("use snb;")

# create graph table
conn.execute("""
CREATE or replace PROPERTY GRAPH snb
VERTEX TABLES (
  Person, Forum
)
EDGE TABLES (
  Person_knows_person     SOURCE KEY (Person1Id) REFERENCES Person (id)
                          DESTINATION KEY (Person2Id) REFERENCES Person (id)
                          LABEL knows,
  Forum_hasMember_Person  SOURCE KEY (ForumId) REFERENCES Forum (id)
                          DESTINATION KEY (PersonId) REFERENCES Person (id)
                          LABEL hasMember
);
""")

result = conn.execute("""
FROM GRAPH_TABLE (snb
  MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:knows]->*(p2:person)
  COLUMNS (p1.id, p2.id as other_person_id, element_id(p), path_length(p))
);
""").fetchall()

for row in result:
    print(row)



# THIS DOES NOT WORK, substrait doesn't support DuckPGQ's custom operators
# result = conn.execute("""CALL get_substrait('
# FROM GRAPH_TABLE (snb
#   MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:knows]->*(p2:person)
#   COLUMNS (p1.id, p2.id as other_person_id, element_id(p), path_length(p))
# )
# ');""")
# print(result)
