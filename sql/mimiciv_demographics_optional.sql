-- Extract optional demographic information for cohort records.
BEGIN;
CREATE SCHEMA IF NOT EXISTS aici_extract;
SET LOCAL search_path = aici_extract, public;
CREATE TABLE ThreeIDTable AS
SELECT DISTINCT subject_id,hadm_id,stay_id
FROM mimic_icu.chartevents;
CREATE TABLE IDAgeGender AS
SELECT
t.subject_id,
t.hadm_id,
t.stay_id,
a.age,
p.gender
FROM ThreeIDTable t
JOIN public.age a
ON t.subject_id=a.subject_id AND t.hadm_id=a.hadm_id
JOIN mimic_hosp.patients p
ON t.subject_id = p.subject_id;
COMMIT;
