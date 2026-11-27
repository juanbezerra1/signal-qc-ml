# Design notes

What was decided, why, and the traps that were met on the way. The README says what the project does; this page says
why it does it that way, and where the first version was wrong.

## The question, and what is the target

*Which recordings will the review want to look at again?* The answer is `flagged` or `rejected` against `passed`.
A recording that has not been reviewed (`pending`) has no answer yet: it is not a class of the target, and a model
that is given it as "not a problem" learns that being new is good. Those rows are left out of every evaluation and are
what `rank` would score.

## Which columns a model may read

A column is one of four things (`columns.py`): who the row is about, the target, something known **before** the review,
or something the review itself produces. The trials that were thrown out are decided in the same review as the
recording, so `n_trials_rejected` and `valid_fraction` are the answer in other words. They stay in the table, and the
evaluation has rows that read them, so that their near-perfect AUC is on record next to the honest numbers and nobody
has to take it on trust that a leak would have been noticed. No set of columns that a *claim* is made about contains
them (`tests/test_evaluate.py` checks that).

The groups are added one at a time (acquisition, session, device, participant, signal), because a number for "all the
columns" alone does not say which part of the table the model leaned on.

## Splitting by participant

The recordings of one participant resemble each other: the same age, the same sessions, often the same device and
operator. A random split puts some of them in training and some in test, and a model that has seen a participant
can recognise the participant instead of the recording. `cv.grouped_splits` keeps every participant in one fold
(`StratifiedGroupKFold`, so that each fold also has about the same share of recordings that need attention), repeats
the cut ten times with other folds, and refuses to go on if a participant is ever on both sides (`LeakageError`).

Each split is also written to two CSV files and given to `table-audit leakage`, a separate project, which looks for a
participant on both sides, for a row that is in both, and for a column that gives the target away. The evaluation
stops if the audit finds an error in any split. The same audit is run on a random split of the same table, and finds
`group-overlap` in every fold: that is what it looks like when it works.

What the audit cannot do is know which columns a review produces (see `docs/columns.md`).

## The AUC of cross-validation is one AUC per fold

The first version pooled the out-of-fold scores of all the folds and computed one AUC. The models of two folds are
different models, so their scores are not on one scale, and a fold that happens to have more recordings that need
attention makes a model that gives lower probabilities to the next one: the pooled AUC of a model that says the same
thing to every recording was 0.48 and 0.49 instead of 0.5. `stats.FoldedAUC` ranks inside each fold and averages the
folds and the passes, which is what scikit-learn's `cross_val_score` does, and which makes the AUC of the prior
(the model that says the share it saw in training to every recording) exactly 0.5. A test checks it against
scikit-learn fold by fold.

## The primary question is stated before the numbers

*Is the AUC of a logistic regression on everything known before the review above what chance gives, for participants
it has not seen?* One model, one set of columns, one statistic (the first pass of the cross-validation), decided before
looking at the results. The other rows of the table (each group of columns, boosting) are there to show where an effect
would come from, not to be searched for one that is significant: with six intervals of 95%, one will be off 0.5 about a
quarter of the time by chance alone.

The reference is a permutation test, and the reference is not 0.5: the AUC of cross-validation on results that mean
nothing is a little above or below it, by an amount that depends on the table. The test moves the results and runs the
whole procedure again, 500 times, and the p-value is where the real AUC falls among them.

### The first way of moving the results was wrong

Moving the results of a participant to another participant who has as many recordings keeps what makes the recordings
of one participant alike (how many of them need attention), which a plain shuffle of single recordings destroys.
The first version moved the results *as they were*, in the order of the recordings. The order says a lot: the two
recordings of a session are an EMG and a force recording, so the position of a recording decides its modality, its
number of channels and its bytes per sample. Moving the results with their order kept whatever chance association
there was between position and result, and the reference was then too high for exactly those columns: on the small
table, a model that read only the acquisition columns had a mean AUC of about 0.57 on tables whose results had been
"moved". `cv.permute_between` now mixes the results among the recordings of the participant that gets them, and the
reference is at 0.50 on all three tables. `test_moving_does_not_carry_the_order_of_the_recordings_along` pins it, and
the mutation check breaks it on purpose.

The assumption that is left: under "nothing predicts it", the recordings of one participant are exchangeable. If a
whole session is bad, they are not, and the reference is a little too narrow.

## Controls

A "no" is worth something only if the same procedure says "yes" when something is there, and says "no" when it is
allowed to cheat only in a way it should refuse.

**The effect that is planted.** A column is added whose value is noise plus `shift` standard deviations for the
recordings that need attention, so that it would have an AUC of Φ(shift/√2) on its own. The evaluation recovers that
AUC (the table in the README has the measured one next to the theoretical one), and the share of tables in which the
primary test sees it says how large an effect the "no" rules out. The first version planted the column in the
*real* results, and the table's own columns have a chance association with them, so with no effect at all the test
"saw" something in a fifth of the tables. The results are now moved between participants first: the only thing left to
find is what was planted, and with a shift of zero the test sees it in about one table in twenty, as it should.

**The leak.** Each participant is given its own chance of having recordings that need attention (the chances come from
a Beta distribution, so a few participants have many and most have none). The columns know nothing about it, but
several of them are the same for all the recordings of a participant, so a model that has seen the participant can
learn the chance. A random split reports an AUC well above 0.5 for boosting and a split by participant does not.

## The same table on every machine

* The database, the seed and the queries are fixed, and the second query summarises the signals with exact decimal
  arithmetic (`numeric`): `avg` on `double precision` changed the sixth decimal between two identical databases,
  because the order in which the rows are added is not fixed.
* The table is rounded to six decimals, sorted by `recording_id`, written as UTF-8 with `\n` line endings.
* The SHA-256 in `data/*.meta.json` is the hash of the CSV **text**, not of the file: a gzipped file is not the same
  bytes on every machine, the text inside it is. `.gitattributes` keeps git from turning the line endings into
  `\r\n` on Windows, which would change the hash.
* `check` (and `evaluate`) refuses a table whose hash is not the one of its provenance file.

## Not done, on purpose

* No tuning of the models. The models have their default settings (a small, slow boosting), so that the evaluation does
  not become a search whose best result is reported.
* `rank` is not gated by the permutation test. 500 refits would make a ranking take minutes, so its gate is the interval
  of the AUC: the floor, which keeps out a model that is plainly no good. An option that `rank` would accept and ignore
  (`--permutations`) is not offered. `evaluate` first, and its primary test.
* No validation by date. A model that is used in a lab sees recordings that are newer than the ones it learnt from,
  of participants it may already know. The evaluation answers a different and harder question (new participants),
  and says so.
* No deep model, no feature selection, no calibration plot: none of them changes the answer when the AUC of the
  columns is at chance.
