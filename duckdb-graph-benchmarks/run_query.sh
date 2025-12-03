#!/bin/bash
set -e

QUERY="query.sql"
SQL_FILE="/data/${QUERY}"
OUT_FILE="/data/plan.dot"

echo "Running DuckDB graph query script..."
duckdb -c ".output $OUT_FILE; SELECT 1;" >/dev/null 2>&1

duckdb <<EOF
.output $OUT_FILE
.read $SQL_FILE
.output stdout
EOF

echo "Done. Output saved to $OUT_FILE"