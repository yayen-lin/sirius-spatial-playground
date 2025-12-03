
SERVICE=duckdb

DB=/data/finbench.duckdb

.PHONY: help
help:
	@echo "Available targets:"
	@echo "  make up         - Start container in background"
	@echo "  make down       - Stop container"
	@echo "  make run        - Run main query script"
	@echo "  make plan       - Export query plan (.dot)"
	@echo "  make shell      - Open duckdb shell interactively"
	@echo "  make sql file=… - Execute arbitrary SQL file"

# Start container in background
.PHONY: up
up:
	docker compose up -d

# Stop docker container and remove volumes
.PHONY: down
down:
	docker compose down -v

# Enter DuckDB CLI interactively
.PHONY: shell
shell:
	docker exec -it $(SERVICE) duckdb $(DB)
	
# Run setup (schema + graph creation)
.PHONY: setup
setup:
	docker exec -it $(SERVICE) duckdb $(DB) -c ".read /scripts/setup.sql"

# Run the SQL script inside the container (non-interactive)
.PHONY: run
run:
	docker exec -it $(SERVICE) duckdb $(DB) -c ".read /scripts/queries.sql"


# Run arbitrary script:   make sql file=myscript.sql
.PHONY: sql
sql:
	docker exec -it $(SERVICE) duckdb $(DB) -c ".read /scripts/$(file)"


# Generate only the EXPLAIN graphviz output as plan.dot
.PHONY: plan
plan:
	docker compose exec -it $(SERVICE) duckdb $(DB) -c ".read /scripts/query1.sql"
	@echo "Plan exported to duckdb_data/finbench_query_graphviz.dot"

# Show container logs (if needed)
.PHONY: logs
logs:
	docker compose logs -f
