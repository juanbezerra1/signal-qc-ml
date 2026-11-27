-- What the signal looked like in each recording: the mean and the spread of every feature of lab.feature_def over the
-- trials and the channels of the recording. Every trial counts, valid or not: whether a trial was thrown out is decided
-- in the review, and the point is to look at what was known before it.
--
-- The arithmetic is done in numeric, not in float8. A float8 average depends on the order in which the rows are added
-- (it differs in the last bits between two servers, or between two runs with parallel workers), and a mean that lands
-- exactly between two roundings then comes out as one number here and another there. In numeric the sum is exact, so
-- two databases with the same data give the same table.
SELECT t.recording_id,
       f.name                                                              AS feature,
       round(avg(tf.value::numeric), 6)::float8                            AS mean,
       round(coalesce(stddev_samp(tf.value::numeric), 0), 6)::float8       AS sd
FROM lab.trial         AS t
JOIN lab.trial_feature AS tf ON tf.trial_id = t.trial_id
JOIN lab.feature_def   AS f  ON f.feature_id = tf.feature_id
GROUP BY t.recording_id, f.name
ORDER BY t.recording_id, f.name;
