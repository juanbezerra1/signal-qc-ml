# The columns of the table

One row per recording. The table is made by the two queries in [`src/signal_qc_ml/`](../src/signal_qc_ml)
(`recordings.sql`, `signals.sql`) and every column belongs to exactly one of the groups below; `columns.py` is the
code version of this page, and `tests/test_columns.py` and `tests/test_docs.py` check that the two agree, and that the
tables in `data/` have these columns and no others.

The evaluation reads the groups **acquisition**, **session**, **device**, **participant** and **signal**: what is
known about a recording before anybody reviews it. It never reads an identifier or `qc`, and it reads the two
**review** columns only in the rows of the results that are there to show what they do.

## Who the row is about (never read by a model)

| Column | What it is |
|---|---|
| `recording_id` | the recording |
| `session_id` | the session it belongs to |
| `participant_id`, `participant_code` | the participant: the unit that is never split between train and test |
| `recorded_on` | the day, in the lab's time zone (America/Sao_Paulo) |

## The target

| Column | What it is |
|---|---|
| `qc` | the quality-control state: `passed`, `flagged`, `rejected`, or `pending` (not reviewed yet). The question is whether a reviewed recording **needs attention**: `flagged` or `rejected`, against `passed`. `pending` is the absence of a result, so those rows are never a class: they are the ones that `rank` scores |

## Known before the review

| Group | Column | What it is |
|---|---|---|
| acquisition | `study_code`, `modality` | the study, and `emg`, `force` or `eeg` |
| acquisition | `block_no`, `sampling_rate_hz`, `n_channels`, `duration_s`, `n_trials` | how the recording was made |
| acquisition | `bytes_per_sample` | file size over duration times rate times channels: close to 4 for a healthy file |
| acquisition | `recorded_hour`, `recorded_weekday` | when, in the lab's clock (weekday 1 is Monday) |
| session | `visit_no`, `session_minutes`, `recordings_in_session` | the session |
| session | `room`, `protocol_version`, `operator`, `operator_role` | where, how and by whom |
| device | `device_id` | the device (a code, not a quantity) |
| device | `device_age_days`, `days_since_calibration`, `calibration_covers` | the device on the day: its age, the days since the latest calibration that passed, and whether one covers the day (0 or 1) |
| participant | `age_years`, `sex`, `dominant_hand`, `group_label` | the participant on the day |
| signal | `sig_<feature>_mean`, `sig_<feature>_sd` | the mean and the spread of each feature of the signal over the trials of the recording: 17 features, 34 columns. A feature that the modality does not have is empty |

## Produced by the review

| Column | What it is |
|---|---|
| `n_trials_rejected`, `valid_fraction` | the trials thrown out are decided in the same review as the recording, so a model that reads them is reading the answer |

They are in the table on purpose: the evaluation shows what a model does with them (a near-perfect AUC), so that the
number is on record next to the honest ones. The audit is given the whole table, review columns included, and does not
call them out: a column produced by the review looks like any other column to a program that does not know what the
review is. What knows it is `columns.py`, which is why the groups are written down there and not inferred.
