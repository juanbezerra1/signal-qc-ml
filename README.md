# signal-qc-ml

[![CI](https://github.com/juanbezerra1/signal-qc-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/juanbezerra1/signal-qc-ml/actions/workflows/ci.yml)

**Does anything that is known before a person reviews a recording say which recordings will need a second look? A small test-bed for asking that without fooling yourself: participants the model has never seen, a test that was stated before the numbers, an effect that is planted to see whether it is found, and a leak that is simulated to see whether it is caught.**

A lab keeps a table of recordings (EMG, force, EEG). A person reviews each one and marks it `passed`, `flagged` or `rejected`.
It is easy to fit a model on that table and report an AUC, and there are three ways for that number to mean nothing: a column
that the review itself produces is in the table, so the model is reading the answer; the recordings of one participant are
on both sides of the split, so the model learns people, not problems; or many models and sets of columns were tried and the
best one is the one that gets reported. `signal-qc-ml` is a pipeline that does none of the three, and a set of controls that
show what its "no" is worth.

The data are synthetic. The tables come from the seed of [labtrack-db](https://github.com/juanbezerra1/labtrack-db), my
PostgreSQL schema for a lab that records EMG, force and EEG, where the result of a review is drawn from a hash of the file path of the
recording, so no column can predict it. The right answer on these tables is therefore "nothing does", which is why they make
a good test: a method that cannot say "no" when there is nothing, or "yes" when there is something, is not worth running on
a real lab. The same code, pointed at a database with the schema of labtrack-db, asks the real question.

Português: [README.pt-BR.md](README.pt-BR.md)

## At a glance

| | |
|---|---:|
| Commands | 5 |
| Recordings in the largest table | 15,008 |
| Participants in the largest table | 1,779 |
| Columns a model may read | 59 |
| Ways the tests are broken on purpose, all noticed | 30 |
| Dependencies | 4, and table-audit for `evaluate` |

If you only have five minutes: read [what it found](#what-it-found), then [what a "no" is worth](#what-a-no-is-worth) and
the [limits](#limits).

```mermaid
flowchart LR
    D["labtrack-db (PostgreSQL)"] --> E["extract: one row per recording, with its SHA-256"]
    E --> T["table: columns known before the review, and the two that the review produces"]
    T --> S["splits by participant, repeated"]
    S --> A["table-audit: no participant on both sides"]
    A --> M["models: prior, logistic, boosting"]
    M --> P["primary test: results moved between participants"]
    M --> C["controls: a planted effect, a simulated leak"]
    P --> R["report, and a ranking only if the model beats chance"]
    C --> R
```

## Quick start

You need Python 3.11 or newer. `evaluate` also needs [table-audit](https://github.com/juanbezerra1/table-audit), another
project of mine, which audits every split before a model sees it.

```bash
git clone https://github.com/juanbezerra1/signal-qc-ml.git
cd signal-qc-ml
py -m pip install git+https://github.com/juanbezerra1/table-audit.git
py -m pip install .
```

On Linux or macOS write `python3 -m pip ...`. The folder [`data/`](data) has three tables that were read from labtrack-db
databases of three sizes. Next to each there is a `.meta.json` with where the table came from and its SHA-256.

### `check`: is this the table that its provenance file describes?

```console
$ py -m signal_qc_ml check data/recordings-small.csv
data/recordings-small.csv: 460 recordings of 59 participants
reviewed 454 (needing attention: 54), pending 6
SHA-256 matches recordings-small.meta.json
```

If the hash does not match, the command stops with exit code `1`, and `evaluate` refuses the table as well.

### `evaluate`: what predicts the review, on participants never seen?

```console
$ py -m signal_qc_ml evaluate data/recordings-medium.csv --out results/medium.json --controls --jobs 2
```

It prints the tables of [what it found](#what-it-found) and writes everything to a JSON file (`report` prints the tables
again from it). With `--controls` it also runs the planted effect and the simulated leak. The medium table took
2 minutes with two processes on the machine where I ran it, the large one 11.
`--skip-audit` runs without table-audit, and the results say that they did.

### `rank`: put the recordings that are waiting for a review in order, if the evidence allows

```console
$ py -m signal_qc_ml rank data/recordings-small.csv --out ranking.csv
signal-qc-ml: refused: the logistic model is not better than chance on participants it has not seen (AUC 0.528, 95% interval 0.463 to 0.585, and a coin gives 0.5), so it would put the recordings in an order that means nothing, and nothing was ranked
```

It evaluates the model again, and only if the 95% interval of its AUC is above 0.5 does it fit it on every reviewed
recording and score the pending ones (exit code `0`, a CSV with `rank, recording_id, participant_code, score`). Otherwise it
stops with exit code `1` and writes nothing, as above: an order that means nothing is worse than no order.

### `extract` and `report`

```console
$ py -m pip install ".[db]"
$ py -m signal_qc_ml extract --dsn postgresql://user:password@localhost:5432/labtrack --out data/recordings.csv
$ py -m signal_qc_ml report results/large.json --section primary --lang en
```

`extract` runs the two queries in [`src/signal_qc_ml/`](src/signal_qc_ml) (`recordings.sql` and `signals.sql`) and writes
the table with its provenance. Exit codes of every command: `0` ok, `1` the tool looked and said no (a leak, a model that
does not beat chance, a table that is not the one it claims to be), `2` the command or an input could not be understood.

## What it found

The question is: among the recordings that have been reviewed, which ones need attention (`flagged` or `rejected`, against
`passed`)? A recording that is still `pending` has no answer yet, so it is not a class of the target and is left out of
every evaluation. The columns are in two groups ([`docs/columns.md`](docs/columns.md) has all of them): the ones known
before anyone looks (how the recording was made, the session, the device, the participant, and the mean and spread of the
signal features) and the two that the review itself produces (`n_trials_rejected` and `valid_fraction`).

The table below is the large run: 15,008 recordings of 1,779 participants,
14,908 of them reviewed and 1,808 of those needing attention
(12.1%). Every row is a model validated on participants it has not seen: five folds, each participant
in one fold only, the whole procedure repeated ten times with other folds. The AUC is the chance that the model puts a
recording that needs attention above one that does not, so 0.5 is a coin. The interval is a bootstrap over participants.

| Columns the model reads | Model | Columns | AUC | 95% interval | Avg. precision | Brier | Known before the review |
| --- | --- | ---: | ---: | --- | ---: | ---: | --- |
| nothing (the share seen in training) | prior | 0 | 0.500 | 0.500 to 0.500 | 0.121 | 0.107 | yes |
| acquisition | logistic | 10 | 0.512 | 0.500 to 0.524 | 0.127 | 0.107 | yes |
| + session | logistic | 17 | 0.514 | 0.502 to 0.525 | 0.127 | 0.107 | yes |
| + device | logistic | 21 | 0.512 | 0.500 to 0.524 | 0.127 | 0.107 | yes |
| + participant | logistic | 25 | 0.510 | 0.498 to 0.522 | 0.126 | 0.107 | yes |
| + signal | logistic | 59 | 0.507 | 0.496 to 0.519 | 0.126 | 0.107 | yes |
| everything known before the review | boosting | 59 | 0.507 | 0.496 to 0.518 | 0.126 | 0.107 | yes |
| only what the review produces | logistic | 2 | 0.995 | 0.994 to 0.997 | 0.981 | 0.011 | no |
| everything known before + what the review produces | logistic | 61 | 0.995 | 0.993 to 0.996 | 0.980 | 0.012 | no |
| everything known before + what the review produces | boosting | 61 | 0.995 | 0.993 to 0.996 | 0.981 | 0.012 | no |

![AUC of every set of columns, on the large table](docs/img/auc.svg)

**What the table says.**

* **Nothing that is known before the review adds up to more than a coin.** The six rows that read only such columns have
  AUCs between 0.507 and 0.514, and they stay there as groups of columns are added. The prior, which
  says the share it saw in training to every recording, is exactly 0.5.
* **The intervals are for sizing an effect, not for deciding that there is one.** They resample participants in scores that
  were already made, and leave out the variation of fitting the model. The lower ends of the first three are above 0.5 by
  less than 0.002 (the table rounds that away), and the three after them reach 0.5. The test below, whose reference does
  include the fitting, is the one that decides.
* **The two columns that the review produces have an AUC of 0.995.** The
  trials that are thrown out are decided in the same review as the recording, so a model that reads them is reading the
  answer. They are in the table on purpose, so that this number is on record next to the honest ones.

The question that was stated before the numbers is one test, not six: does a logistic regression on everything known before
the review have an AUC above what it has when the results mean nothing? Its statistic is the first pass of the
cross-validation, and the reference is 500 versions of the table in which the results of whole
participants were moved to other participants with the same number of recordings (and mixed among the recordings of the
participant that gets them: moving them intact carries the order of the recordings along, and the reference was wrong
until that was found, see [`docs/design-notes.md`](docs/design-notes.md)).

| AUC of the first pass | Mean when results are moved | Spread (sd) | 95th percentile | p-value | Moves |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.506 | 0.500 | 0.010 | 0.516 | 0.273 | 500 |

The p-value is 0.273: that share of the moved tables does at least as well as the real one. Six intervals
of 95% that share most of their columns can leave some of them off 0.5 by chance alone, which is why the other rows are for
looking and this one is for deciding.

## What a "no" is worth

A "no" from a method that could not have said "yes" is not information. Two controls are run on every table.

### An effect that is planted

A column of noise is added whose value is shifted by a number of standard deviations for the recordings that need
attention, so that alone it would have a known AUC. It is added after the results were moved between participants, so that
it is the only thing left to find, and the primary test is run again, 20 times for every shift. The last column is the share
of those tables in which the test saw the effect (an AUC above the 95th percentile of the moved ones). With a shift of zero
that should be about one in twenty.

| Planted shift (sd) | AUC of the column (theory) | AUC of the column (measured) | AUC with everything known | Seen by the primary test |
| ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0.500 | 0.497 | 0.500 | 0% (20) |
| 0.1 | 0.528 | 0.532 | 0.516 | 50% (20) |
| 0.2 | 0.556 | 0.556 | 0.538 | 100% (20) |
| 0.3 | 0.584 | 0.584 | 0.567 | 100% (20) |
| 0.4 | 0.611 | 0.612 | 0.599 | 100% (20) |
| 0.5 | 0.638 | 0.638 | 0.626 | 100% (20) |
| 0.6 | 0.664 | 0.664 | 0.655 | 100% (20) |
| 0.8 | 0.714 | 0.713 | 0.705 | 100% (20) |

![Share of planted effects that the test sees, on three table sizes](docs/img/power.svg)

On the large table the weakest effect that was seen at least 80% of the time has a shift of 0.2, an AUC of
0.556 for the column alone. So the "no" above rules out an effect of about that size or larger, and says
nothing about smaller ones. How much a "no" can rule out depends on the size of the table: on the medium table the
weakest effect that is seen that often has an AUC of 0.638 (a shift of 0.5), and on the small
table the largest one that was tried, 0.714 for the column alone, was seen 60% of the time. The
figure has the three curves.

### A leak that is simulated

Every participant is given a tendency of their own to have recordings that need attention (a Beta distribution, with a
correlation of 0.23 between two recordings of the same participant). Nothing in the columns says what it is, but several
columns have the same value in all the recordings of a participant, so a model that has seen the participant can learn the
tendency. A split at random lets it, and a split by participant does not:

| Split | AUC of boosting |
| --- | ---: |
| At random (ignores participants) | 0.533 |
| By participant | 0.499 |

![AUC of boosting with a split at random and with a split by participant](docs/img/leak.svg)

With a random split the model looks as if it knew something. With a split by participant, which is the question that is
being asked, it does not: the first number is real, and its conclusion is not.

In 10 of the 10 simulations the random split reported more than the split
by participant. The gap is smaller on the large table than on the other two (the figure has all three), and I did not test
why.

### The audit

Every split of the evaluation (50 per table) is written as a training file and a test file and given to
`table-audit leakage`, which looks for a participant on both sides, for a row that is on both sides, and for a column that
gives the target away. The evaluation stops if it finds an error. The same audit on five random splits finds
`group-overlap` in all five, which is what it looks like when it works:

|  | Splits audited | Splits with errors | Checks that fired (splits) |
| --- | ---: | ---: | --- |
| By participant | 50 | 0 | none |
| At random | 5 | 5 | group-overlap (5) |

## Limits

* **The data are synthetic, and the answer was built in.** The seed of labtrack-db makes a recording `passed` (88%),
  `flagged` (7%) or `rejected` (5%) from a hash of its file path, unless it is recent, and then it is `pending`. This
  project shows that the machinery says "no" when there is nothing, "yes" when something is planted, and catches a leak. It
  does not say anything about a lab.
* **It validates by participant, not by date.** A model that is used in a lab sees recordings that are newer than the ones it
  learned from, of participants that it may know. This evaluation asks the harder question (new people) and has nothing to
  say about the easier one (new recordings of people it knows).
* **The intervals leave out the model.** They resample participants in the scores that were already made; they do not include
  the variation of fitting the model on another sample.
* **`rank` is gated by the interval, which is the weak test.** It refuses a model whose interval reaches 0.5, which keeps
  out the models that are plainly no good, but a model that passes has not been through the permutation test. Run
  `evaluate` first and read its primary test before trusting an order.
* **The permutation test assumes that, when nothing predicts the result, the recordings of one participant can be swapped
  for one another.** If a whole session is bad, they cannot, and the reference is a little too narrow.
* **The primary test was fixed in the code before the final runs, but I had looked at the small table while building it.**
  It is not a pre-registration.
* **The models are not tuned.** Default settings, a small boosting. With nothing to find, tuning is only a search.
* **The audit does not know which columns the review produces.** It looks at the table it is given; `columns.py` is what
  knows, and it is why those two columns never enter a claim.
* **Twenty tables per planted effect is few.** A share of 80% seen has a margin of about plus or minus 18 points, so the
  smallest effect that is "seen" is where the curve crosses the line, not a sharp edge.
* **The numbers belong to a version of scikit-learn.** With the same seed, `StratifiedGroupKFold` put other participants in
  each fold before version 1.8 (I compared 1.7.0 and 1.7.2 with 1.8.0, 1.9.0 and 1.9.1), so `pyproject.toml` asks for 1.8
  or later. The files in `results/` record the versions that made them (scikit-learn 1.9.1); the tests that make a row of
  them again pass with 1.8.0 and with 1.9.1.
* **Not run by me:** the Windows job and GitHub Actions (the workflow is written; the first run is theirs), and a release to
  PyPI (there is none: install from the folder).

## How it is checked

* **Tests** (`py -m pytest`): the AUC against scikit-learn fold by fold, ties and weights against a brute-force count, the
  splits against a participant on both sides, the null against a table with nothing in it and against one with an effect,
  the commands and their exit codes, the two SQL queries on a real PostgreSQL (skipped if there is none), the results
  files against the tables in `data/`, and this README against `results/*.json` and the code.
* **Mutations** (`py scripts/mutation_check.py`): 30 small ways of breaking the code on purpose (a tie between two scores
  that counts as a loss, a split that forgets the participants, a ranking that puts the least likely recording first), each of
  which at least one test must notice. The list itself is checked: every change has to apply exactly once.
* **Types and style:** `ruff`, `ruff format --check` and `mypy` in strict mode.
* **CI:** Python 3.11 to 3.14 on Linux, 3.13 on Windows, the oldest versions of the dependencies that `pyproject.toml`
  allows, and the mutation check. The README and the figures are regenerated in CI and compared with what is in the
  repository.
* **What I ran:** on Linux, the tests with Python 3.11, 3.12, 3.13 and 3.14, and on 3.11 with the oldest versions that
  `pyproject.toml` allows; the mutation check, `ruff` and `mypy` with 3.13; the database tests against a local PostgreSQL 16.

## Files

| Path | What it is |
|---|---|
| [`src/signal_qc_ml/columns.py`](src/signal_qc_ml/columns.py) | which column is what: identifier, target, known before the review, produced by it |
| [`src/signal_qc_ml/extract.py`](src/signal_qc_ml/extract.py), `recordings.sql`, `signals.sql` | the table from a database, and its provenance |
| [`src/signal_qc_ml/cv.py`](src/signal_qc_ml/cv.py) | splits by participant, `LeakageError`, out-of-fold scores, moving the results |
| [`src/signal_qc_ml/stats.py`](src/signal_qc_ml/stats.py) | the AUC, the AUC per fold, the bootstrap over participants |
| [`src/signal_qc_ml/evaluate.py`](src/signal_qc_ml/evaluate.py) | the rows of the table and the primary test |
| [`src/signal_qc_ml/controls.py`](src/signal_qc_ml/controls.py) | the planted effect and the simulated leak |
| [`src/signal_qc_ml/audit.py`](src/signal_qc_ml/audit.py) | every split to table-audit |
| [`src/signal_qc_ml/rank.py`](src/signal_qc_ml/rank.py) | the ranking, and when it refuses |
| [`data/`](data), [`results/`](results) | the three tables with their provenance, and what was found on them |
| [`docs/design-notes.md`](docs/design-notes.md) | what was decided, why, and where the first version was wrong |
| [`scripts/`](scripts) | the mutation check, the figures, this README |

## License

MIT. See [LICENSE](LICENSE).
