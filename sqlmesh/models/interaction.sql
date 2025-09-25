MODEL(
name sqlmesh.interaction,
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
	sale_price_log,
	gr_liv_area * overall_qual AS qual_livarea_interaction
FROM sqlmesh.log_saleprice

