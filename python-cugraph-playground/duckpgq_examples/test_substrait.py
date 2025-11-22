import json
import duckdb

conn = duckdb.connect(':memory:')

# Install and load substrait
conn.execute("INSTALL substrait FROM community;")
conn.execute("LOAD substrait;")

# Test: Generate a Substrait plan from a simple query
conn.execute("CREATE TABLE test (id INT, name VARCHAR);")
conn.execute("INSERT INTO test VALUES (1, 'Alice'), (2, 'Bob');")

# Get Substrait plan
result = conn.execute("CALL get_substrait_json('SELECT * FROM test WHERE id = 1')").fetchone()
substrait_json = result[0]
print(json.dumps(json.loads(substrait_json), indent=4))
