AUDIT (
  name audit_gr_liv_area,
);

SELECT * AS num_outliers_remaining
FROM sqlmesh.drop_gr_liv_area
WHERE sale_price < 200000
  AND overall_qual > 8
  AND gr_liv_area > 4000;

