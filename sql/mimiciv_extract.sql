-- Merge clinical records and extract sepsis and non-sepsis cohorts.
BEGIN;
CREATE SCHEMA IF NOT EXISTS aici_extract;
SET LOCAL search_path = aici_extract, public;
CREATE TABLE table0 (
    subject_id INTEGER,
    stay_id INTEGER,
    charttime TIMESTAMP,
    itemid INTEGER,
    value TEXT
);
INSERT INTO table0 (subject_id, stay_id, charttime, itemid, value)
SELECT subject_id, stay_id, charttime, itemid, value
FROM (
    SELECT subject_id, stay_id, charttime, itemid, value FROM output2
    UNION ALL
    SELECT subject_id, stay_id, charttime, itemid, value FROM gcs1
        UNION ALL
    SELECT subject_id, stay_id, charttime, itemid, value FROM chart1
        UNION ALL
    SELECT subject_id, stay_id, charttime, itemid, value FROM vital1
) AS combined
ORDER BY subject_id, stay_id, charttime;
WITH TimeRange AS (
    SELECT
        subject_id,
        stay_id,
        MIN(charttime) AS min_charttime,
        MAX(charttime) AS max_charttime
    FROM
        table0
    GROUP BY
        subject_id,
        stay_id
)
INSERT INTO table0 (subject_id, stay_id, charttime, itemid, value)
SELECT
    i.subject_id,
    i.stay_id,
    i.charttime,
    i.itemid,
    i.value
FROM
    input11 i
INNER JOIN TimeRange tr ON i.subject_id = tr.subject_id AND i.stay_id = tr.stay_id
WHERE
    i.charttime BETWEEN tr.min_charttime AND tr.max_charttime;
CREATE TABLE table2_no AS
WITH A_Aggregated AS (
    SELECT
        subject_id,
        stay_id,
        MIN(charttime) AS min_charttime,
        MAX(charttime) AS max_charttime
    FROM table0
    GROUP BY subject_id, stay_id
        )
SELECT
    A_Aggregated.subject_id,
        A_Aggregated.stay_id,
        A_Aggregated.min_charttime,
        A_Aggregated.max_charttime,
        A_Aggregated.max_charttime-A_Aggregated.min_charttime AS time_diff
FROM
    A_Aggregated
            LEFT JOIN public.sepsis3 B ON A_Aggregated.subject_id = B.subject_id AND A_Aggregated.stay_id = B.stay_id
WHERE
    (
        B.subject_id IS NULL OR
        B.stay_id IS NULL OR
        A_Aggregated.subject_id != B.subject_id OR
        A_Aggregated.stay_id != B.stay_id
    )
    AND
    A_Aggregated.max_charttime >= A_Aggregated.min_charttime + interval '8 hours';
CREATE TABLE table3_no AS
SELECT c.subject_id,
       c.stay_id,
       c.charttime,
             s.max_charttime,
       c.itemid,
       c.value
FROM table2_no s
JOIN table0 c
ON s.subject_id = c.subject_id AND s.stay_id = c.stay_id;
CREATE TABLE table1 AS
WITH A_Aggregated AS (
    SELECT
        subject_id,
        stay_id,
        MIN(charttime) AS min_charttime,
        MAX(charttime) AS max_charttime
    FROM table0
    GROUP BY subject_id, stay_id
)
    SELECT
        A_Aggregated.subject_id,
        A_Aggregated.stay_id,
        A_Aggregated.min_charttime,
        A_Aggregated.max_charttime,
        B.sofa_time
    FROM A_Aggregated
    LEFT JOIN public.sepsis3 B ON A_Aggregated.subject_id = B.subject_id AND A_Aggregated.stay_id = B.stay_id
    WHERE B.sofa_time >= A_Aggregated.min_charttime + interval '8 hours';
CREATE TABLE table2 AS
WITH A_Aggregated AS (
    SELECT
        c.subject_id,
        c.stay_id,
                c.charttime,
                s.sofa_time
    FROM table1 s
        JOIN table0 c ON c.subject_id=s.subject_id AND c.stay_id=s.stay_id
        WHERE c.charttime<=s.sofa_time
    GROUP BY c.subject_id, c.stay_id,c.charttime,s.sofa_time
),
A_filter AS (
SELECT
    subject_id,
    stay_id,
        MAX(charttime) AS max_charttime,
    sofa_time
        FROM
        A_Aggregated
        GROUP BY subject_id, stay_id,sofa_time
)
SELECT
    subject_id,
    stay_id,
        max_charttime,
    sofa_time,
        sofa_time-max_charttime AS max_diff
FROM A_filter
WHERE sofa_time <= max_charttime + interval '24 hours'
ORDER BY max_diff;
CREATE TABLE table3 AS
SELECT c.subject_id,
       c.stay_id,
       c.charttime,
             s.sofa_time,
       c.itemid,
       c.value
FROM table2 s
JOIN table0 c
ON s.subject_id = c.subject_id AND s.stay_id = c.stay_id
WHERE sofa_time >= charttime;
COMMIT;
