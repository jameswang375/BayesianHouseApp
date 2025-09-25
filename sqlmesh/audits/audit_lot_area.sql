AUDIT(
name audit_lot_area,
);

SELECT * AS num_outliers_remaining
FROM sqlmesh.drop_lot_area
WHERE sale_price <= 400000 AND lot_area > 100000

