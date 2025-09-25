MODEL (
    name sqlmesh.standardized_model,
    kind FULL,
    grain (PID),
    allow_partials true,
    audits(audit_standardization)
);


SELECT
PID,

( "gr_liv_area" - AVG("gr_liv_area") OVER () ) / STDDEV_POP("gr_liv_area") OVER () AS gr_liv_area_std,
    ( "first_flr_sf" - AVG("first_flr_sf") OVER () ) / STDDEV_POP("first_flr_sf") OVER () AS first_flr_sf_std,
    ( "lot_area" - AVG("lot_area") OVER () ) / STDDEV_POP("lot_area") OVER () AS lot_area_std,
    ( "garage_area" - AVG("garage_area") OVER () ) / STDDEV_POP("garage_area") OVER () AS garage_area_std,
    ( "total_bsmt_sf" - AVG("total_bsmt_sf") OVER () ) / STDDEV_POP("total_bsmt_sf") OVER () AS total_bsmt_sf_std,
    ( "qual_livarea_interaction" - AVG("qual_livarea_interaction") OVER () ) / STDDEV_POP("qual_livarea_interaction") OVER () AS qual_livarea_interaction_std,

CASE WHEN overall_qual = 1 THEN 1 ELSE 0 END AS overall_qual_1,
CASE WHEN overall_qual = 2 THEN 1 ELSE 0 END AS overall_qual_2,
CASE WHEN overall_qual = 3 THEN 1 ELSE 0 END AS overall_qual_3,
CASE WHEN overall_qual = 4 THEN 1 ELSE 0 END AS overall_qual_4,
CASE WHEN overall_qual = 5 THEN 1 ELSE 0 END AS overall_qual_5,
CASE WHEN overall_qual = 6 THEN 1 ELSE 0 END AS overall_qual_6,
CASE WHEN overall_qual = 7 THEN 1 ELSE 0 END AS overall_qual_7,
CASE WHEN overall_qual = 8 THEN 1 ELSE 0 END AS overall_qual_8,
CASE WHEN overall_qual = 9 THEN 1 ELSE 0 END AS overall_qual_9,
CASE WHEN overall_qual = 10 THEN 1 ELSE 0 END AS overall_qual_10,

("year_built" - AVG("year_built") OVER () ) / STDDEV_POP("year_built") OVER () AS year_built_std,
("year_remod_add" - AVG("year_remod_add") OVER () ) / STDDEV_POP("year_remod_add") OVER () AS year_remod_add_std,

CASE WHEN overall_cond = 1 THEN 1 ELSE 0 END AS overall_cond_1,
CASE WHEN overall_cond = 2 THEN 1 ELSE 0 END AS overall_cond_2,
CASE WHEN overall_cond = 3 THEN 1 ELSE 0 END AS overall_cond_3,
CASE WHEN overall_cond = 4 THEN 1 ELSE 0 END AS overall_cond_4,
CASE WHEN overall_cond = 5 THEN 1 ELSE 0 END AS overall_cond_5,
CASE WHEN overall_cond = 6 THEN 1 ELSE 0 END AS overall_cond_6,
CASE WHEN overall_cond = 7 THEN 1 ELSE 0 END AS overall_cond_7,
CASE WHEN overall_cond = 8 THEN 1 ELSE 0 END AS overall_cond_8,
CASE WHEN overall_cond = 9 THEN 1 ELSE 0 END AS overall_cond_9,

CASE WHEN full_bath = 0 THEN 1 ELSE 0 END AS full_bath_0,
CASE WHEN full_bath = 1 THEN 1 ELSE 0 END AS full_bath_1,
CASE WHEN full_bath = 2 THEN 1 ELSE 0 END AS full_bath_2,
CASE WHEN full_bath = 3 THEN 1 ELSE 0 END AS full_bath_3,
CASE WHEN full_bath = 4 THEN 1 ELSE 0 END AS full_bath_4,

("sale_price_log" - AVG("sale_price_log") OVER () ) / STDDEV_POP("sale_price_log") OVER () AS sale_price_log_std

FROM sqlmesh.interaction


