AUDIT (name no_nulls);

SELECT * 
FROM sqlmesh.drop_null
WHERE garage_area IS NULL OR total_bsmt_sf IS NULL;