MODEL(
name sqlmesh.log_saleprice,
kind FULL,
grain PID,
allow_partials true);

SELECT
	PID,
	overall_qual,
	gr_liv_area,
	first_flr_sf,
	year_built,
	year_remod_add,
	lot_area,
	overall_cond,
	garage_area,
	total_bsmt_sf,
	full_bath,
	LN(sale_price) AS sale_price_log
FROM sqlmesh.drop_lot_area




