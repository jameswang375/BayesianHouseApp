MODEL(
name sqlmesh.drop_gr_liv_area,
kind FULL,
grain PID,
allow_partials true,
audits(audit_gr_liv_area));

SELECT * 
FROM sqlmesh.drop_null
WHERE NOT (
sale_price < 200000 AND overall_qual > 8 AND gr_liv_area > 4000
)
