
-- Attach the FinBench dataset
-- ATTACH IF NOT EXISTS 'https://blobs.duckdb.org/data/finbench.duckdb' AS finbench;
ATTACH IF NOT EXISTS 'finbench.duckdb' AS finbench;
USE finbench;


-- -- Optionally show counts
-- SELECT
--     (SELECT count(*) FROM Person) AS num_persons,
--     (SELECT count(*) FROM Account) AS num_accounts,
--     (SELECT count(*) FROM AccountTransferAccount) AS num_transfers;

-- Install and load DuckPGQ
INSTALL duckpgq FROM community;
LOAD duckpgq;

-- Create the property graph
-- CREATE PROPERTY GRAPH finbench
-- VERTEX TABLES (
--     Person,
--     Account
-- )
-- EDGE TABLES (
--     AccountTransferAccount
--         SOURCE KEY (fromId) REFERENCES Account (accountId)
--         DESTINATION KEY (toId) REFERENCES Account (accountId)
--         LABEL Transfer,
--     PersonOwnAccount
--         SOURCE KEY (personId) REFERENCES Person (personId)
--         DESTINATION KEY (accountId) REFERENCES Account (accountId)
--         LABEL PersonOwn
-- );