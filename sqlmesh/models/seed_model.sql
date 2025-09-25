MODEL (
name sqlmesh.seed_model,
kind SEED(
	path '../seeds/AmesHousingData.csv'
),
grain PID,
allow_partials true
);
