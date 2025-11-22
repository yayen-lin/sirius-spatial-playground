Set up
```sql
ATTACH 'https://github.com/Dtenwolde/duckpgq-docs/raw/refs/heads/main/datasets/snb.duckdb';

use snb;
install duckpgq from community; 
load duckpgq;

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
```

Shortest Path
```sql
-- find the shortest path from one person to all other persons
FROM GRAPH_TABLE (snb
  MATCH p = ANY SHORTEST (p1:person WHERE p1.id = 14)-[k:knows]->*(p2:person)
  COLUMNS (p1.id, p2.id as other_person_id, element_id(p), path_length(p))
);
```

Mutual Friends
```sql
-- Find mutual friends between two users
FROM GRAPH_TABLE (snb
  MATCH (p1:Person WHERE p1.id = 16)-[k:knows]->(p2:Person)<-[k2:knows]-(p3:Person WHERE p3.id = 32)
  COLUMNS (p2.firstName)
);
```