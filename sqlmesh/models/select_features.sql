MODEL(
name sqlmesh.select_features,
kind FULL,
grain PID,
allow_partials true);

SELECT 
	PID,
	"Overall Qual" AS overall_qual,
	"Gr Liv Area" AS gr_liv_area,
	"1st Flr SF" AS first_flr_sf,
	"Year Built" AS year_built,
	"Year Remod/Add" AS year_remod_add,
	"Lot Area" AS lot_area,
	"Overall Cond" AS overall_cond,
	"Garage Area" AS garage_area,
	"Total Bsmt SF" AS total_bsmt_sf,
	"Full Bath" AS full_bath,
	"SalePrice" AS sale_price
FROM sqlmesh.seed_model

