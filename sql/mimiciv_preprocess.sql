-- Preprocess MIMIC-IV clinical records into intermediate tables.
BEGIN;
CREATE SCHEMA IF NOT EXISTS aici_extract;
SET LOCAL search_path = aici_extract, public;
CREATE TABLE chart1 AS
    SELECT
        A.subject_id,
        A.stay_id,
        A.charttime,
                A.itemid,
                A.value
    FROM mimic_icu.chartevents A
    JOIN mimic_icu.d_items ON A.itemid = mimic_icu.d_items.itemid
    WHERE
        A.itemid = ANY(ARRAY[
223835,
220224,
220228,
220235,
220545,
220546,
220587,
220602,
220615,
220621,
220635,
220644,
220645,
223830,
224828,
225612,
225624,
225625,
225664,
225667,
225668,
225677,
225690,
225698,
226537,
227073,
227429,
227442,
227443,
227457,
227464,
227465,
227466,
227467,
223835,
227456,
220074]);
CREATE TABLE output1 (
    subject_id INTEGER,
    stay_id INTEGER,
    charttime TIMESTAMP,
    itemid INTEGER,
    value TEXT
);
INSERT INTO output1 (subject_id, stay_id, charttime, itemid, value)
SELECT subject_id, stay_id, charttime, itemid, value::text
FROM mimic_icu.outputevents
ORDER BY subject_id ASC, stay_id ASC;
CREATE TABLE output2 AS
SELECT
    subject_id,
    stay_id,
    (DATE_TRUNC('HOUR', charttime)) as charttime,
    260000 as itemid,
    CAST(SUM(CAST(value AS FLOAT)) AS TEXT) as value
FROM
    output1
WHERE
    itemid IN (226559, 226560)
GROUP BY
    subject_id,
    stay_id,
    DATE_TRUNC('HOUR', charttime);
CREATE TABLE gcs1 (
    subject_id INTEGER,
    stay_id INTEGER,
    charttime TIMESTAMP,
    itemid INTEGER,
    value TEXT
);
INSERT INTO gcs1 (subject_id, stay_id, charttime, itemid, value)
SELECT
    subject_id,
    stay_id,
    charttime,
    240001 AS itemid,
    gcs::text AS value
FROM
    public.gcs
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    240002 AS itemid,
    gcs_motor::text AS value
FROM
    public.gcs
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    240003 AS itemid,
    gcs_verbal::text AS value
FROM
    public.gcs
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    240004 AS itemid,
    gcs_eyes::text AS value
FROM
    public.gcs
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    240005 AS itemid,
    gcs_unable::text AS value
FROM
    public.gcs
ORDER BY subject_id, stay_id, charttime;
CREATE TABLE vital AS
SELECT ce.subject_id,
    ce.stay_id,
    ce.charttime,
    avg(
        CASE
            WHEN ((ce.itemid = 220045) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (300)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS heart_rate,
    avg(
        CASE
            WHEN ((ce.itemid = ANY (ARRAY[220179, 220050,225309])) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (400)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS sbp,
    avg(
        CASE
            WHEN ((ce.itemid = ANY (ARRAY[220180, 220051,225310])) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (300)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS dbp,
    avg(
        CASE
            WHEN ((ce.itemid = ANY (ARRAY[220052, 220181, 225312])) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (300)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS mbp,
    avg(
        CASE
            WHEN ((ce.itemid = 220179) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (400)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS sbp_ni,
    avg(
        CASE
            WHEN ((ce.itemid = 220180) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (300)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS dbp_ni,
    avg(
        CASE
            WHEN ((ce.itemid = 220181) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (300)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS mbp_ni,
    avg(
        CASE
            WHEN ((ce.itemid = ANY (ARRAY[220210, 224690])) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum < (70)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS resp_rate,
    round((avg(
        CASE
            WHEN ((ce.itemid = 223761) AND (ce.valuenum > (70)::double precision) AND (ce.valuenum < (120)::double precision)) THEN ((ce.valuenum - (32)::double precision) / (1.8)::double precision)
            WHEN ((ce.itemid = 223762) AND (ce.valuenum > (10)::double precision) AND (ce.valuenum < (50)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END))::numeric, 2) AS temperature,
    max((
        CASE
            WHEN (ce.itemid = 224642) THEN ce.value
            ELSE NULL::character varying
        END)::text) AS temperature_site,
    avg(
        CASE
            WHEN ((ce.itemid = 220277) AND (ce.valuenum > (0)::double precision) AND (ce.valuenum <= (100)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS spo2,
    avg(
        CASE
            WHEN ((ce.itemid = ANY (ARRAY[225664, 220621, 226537])) AND (ce.valuenum > (0)::double precision)) THEN ce.valuenum
            ELSE NULL::double precision
        END) AS glucose
   FROM mimic_icu.chartevents ce
  WHERE ((ce.stay_id IS NOT NULL) AND (ce.itemid = ANY (ARRAY[220045, 225309, 225310, 225312, 220050, 220051, 220052, 220179, 220180, 220181, 220210, 224690, 220277, 225664, 220621, 226537, 223762, 223761, 224642])))
  GROUP BY ce.subject_id, ce.stay_id, ce.charttime;
CREATE TABLE vital1 (
    subject_id INTEGER,
    stay_id INTEGER,
    charttime TIMESTAMP,
    itemid INTEGER,
    value TEXT
);
INSERT INTO vital1 (subject_id, stay_id, charttime, itemid, value)
SELECT
    subject_id,
    stay_id,
    charttime,
    250001 AS itemid,
    heart_rate::text AS value
FROM
    vital
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    250002 AS itemid,
    sbp::text AS value
FROM
    vital
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    250003 AS itemid,
    dbp::text AS value
FROM
    vital
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    250004 AS itemid,
    mbp::text AS value
FROM
    vital
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    250005 AS itemid,
    resp_rate::text AS value
FROM
    vital
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    250006 AS itemid,
    temperature::text AS value
FROM
    vital
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    250007 AS itemid,
    spo2::text AS value
FROM
    vital
UNION ALL
SELECT
    subject_id,
    stay_id,
    charttime,
    250008 AS itemid,
    glucose::text AS value
FROM
    vital
ORDER BY subject_id, stay_id, charttime;
CREATE TABLE input11 (
    subject_id INTEGER,
    stay_id INTEGER,
    charttime TIMESTAMP,
    itemid INTEGER,
    value TEXT
);
INSERT INTO input11 (subject_id, stay_id, charttime, itemid, value)
SELECT subject_id, stay_id, charttime, itemid, amount::text
FROM public.input1
WHERE itemid = ANY (ARRAY[225158,220949,226452,225943,225166,227522]);
COMMIT;
