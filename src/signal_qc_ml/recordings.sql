-- One row per recording: what is known about it, how it was recorded, and its quality-control state.
--
-- Three kinds of column come out of this query (see columns.py):
--   * who the row is about: recording_id, session_id, participant_id, participant_code, recorded_on;
--   * what is known BEFORE anyone reviews the recording: how it was recorded, the session, the device and its
--     calibration on the day, the participant;
--   * what the review itself produces: n_trials_rejected and valid_fraction (rejected trials are decided in the same
--     review as the recording, so a model that reads them is reading the answer), and qc, the thing to predict.
--
-- The lab's clock is America/Sao_Paulo: "the day" and "the hour" are the lab's, not the server's.
WITH base AS (
    SELECT recording_id,
           recorded_at AT TIME ZONE 'America/Sao_Paulo' AS local_at,
           (recorded_at AT TIME ZONE 'America/Sao_Paulo')::date AS day
    FROM lab.recording
),
calibration AS (
    -- the latest calibration that passed on or before the day of each recording, and whether one covers that day
    SELECT b.recording_id,
           max(c.performed_on) FILTER (WHERE c.passed AND c.performed_on <= b.day) AS last_passed_on,
           coalesce(bool_or(c.passed AND c.performed_on <= b.day AND b.day < c.valid_until), false) AS covered
    FROM base AS b
    JOIN lab.recording AS r ON r.recording_id = b.recording_id
    LEFT JOIN lab.calibration AS c ON c.device_id = r.device_id
    GROUP BY b.recording_id
),
trials AS (
    SELECT recording_id,
           count(*) AS n_trials,
           count(*) FILTER (WHERE NOT valid) AS n_trials_rejected
    FROM lab.trial
    GROUP BY recording_id
),
in_session AS (
    SELECT session_id, count(*) AS recordings_in_session
    FROM lab.recording
    GROUP BY session_id
)
SELECT rec.recording_id,
       rec.session_id,
       s.participant_id,
       p.code                                             AS participant_code,
       b.day                                              AS recorded_on,
       -- known before the review: the recording
       st.code                                            AS study_code,
       rec.modality::text                                 AS modality,
       rec.block_no::int                                  AS block_no,
       rec.sampling_rate_hz,
       rec.n_channels::int                                AS n_channels,
       rec.duration_s::float8                             AS duration_s,
       coalesce(t.n_trials, 0)                            AS n_trials,
       rec.size_bytes::float8 / nullif(rec.duration_s * rec.sampling_rate_hz * rec.n_channels, 0)
                                                          AS bytes_per_sample,
       extract(hour FROM b.local_at)::int                 AS recorded_hour,
       extract(isodow FROM b.local_at)::int               AS recorded_weekday,
       -- known before the review: the session
       s.visit_no::int                                    AS visit_no,
       s.room,
       s.protocol_version,
       s.duration_min                                     AS session_minutes,
       n.recordings_in_session,
       s.researcher_id                                    AS operator,
       rs.role::text                                      AS operator_role,
       -- known before the review: the device
       rec.device_id,
       (b.day - d.acquired_on)                            AS device_age_days,
       (b.day - c.last_passed_on)                         AS days_since_calibration,
       c.covered::int                                     AS calibration_covers,
       -- known before the review: the participant
       (extract(year FROM b.local_at) - p.birth_year)::int AS age_years,
       p.sex::text                                        AS sex,
       p.dominant_hand::text                              AS dominant_hand,
       e.group_label,
       -- produced by the review
       coalesce(t.n_trials_rejected, 0)                   AS n_trials_rejected,
       1 - t.n_trials_rejected::float8 / nullif(t.n_trials, 0) AS valid_fraction,
       rec.qc::text                                       AS qc
FROM lab.recording AS rec
JOIN base          AS b  ON b.recording_id = rec.recording_id
JOIN lab.session   AS s  ON s.session_id = rec.session_id
JOIN lab.study     AS st ON st.study_id = s.study_id
JOIN lab.participant AS p ON p.participant_id = s.participant_id
JOIN lab.enrollment  AS e ON e.study_id = s.study_id AND e.participant_id = s.participant_id
JOIN lab.researcher  AS rs ON rs.researcher_id = s.researcher_id
JOIN lab.device      AS d ON d.device_id = rec.device_id
JOIN calibration     AS c ON c.recording_id = rec.recording_id
JOIN in_session      AS n ON n.session_id = rec.session_id
LEFT JOIN trials     AS t ON t.recording_id = rec.recording_id
ORDER BY rec.recording_id;
