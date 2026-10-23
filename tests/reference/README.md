# Frozen pre-refactor references

`legacy/` contains byte-identical KneeJerk research modules and the dependency
closure required by the parity tests. They are comparison fixtures, not supported
training commands. Production Bazza modules never import this directory.
Original files and SHA-256 hashes are recorded in `docs/MIGRATION_SOURCE.json`.
The complete research history remains in KneeJerk; unrelated experiments and
frontend prototypes are not part of this repository.
