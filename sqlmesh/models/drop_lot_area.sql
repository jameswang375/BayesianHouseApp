MODEL(
name sqlmesh.drop_lot_area,
kind FULL,
grain PID,
allow_partials true,
audits(audit_lot_area)
);

SELECT *
FROM sqlmesh.drop_gr_liv_area
WHERE NOT (
sale_price <= 400000 AND lot_area > 100000
)
