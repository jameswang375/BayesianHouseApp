AUDIT (
  name audit_standardization,

);

SELECT *
FROM sqlmesh.standardized_model
WHERE ABS(gr_liv_area_std) > 1e6
