MODEL(
name sqlmesh.drop_null,
kind FULL,
grain PID,
allow_partials true,
audits (no_nulls) );

SELECT * 
FROM sqlmesh.select_features
WHERE garage_area IS NOT NULL AND total_bsmt_sf IS NOT NULL
