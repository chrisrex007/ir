# Programming Assignment III: Learning to Rank

**Group:** search_ninjas

**Collection:** Cranfield (1,400 documents; 225 queries)

**Language:** Python 3.12

This assignment compares weighted Boolean zone scoring with a linear support
vector machine for document ranking. The code trains both methods, evaluates
them on a fixed query split, and produces TREC-format ranking files and a PDF
report. The five members' names and roll numbers are in
`search_ninjas_group.py` and on the report's first page.

## Setup and evaluation

Run these commands from the `asgn3` directory:

```bash
python3.12 -m venv .venv
.venv/bin/python -m ensurepip --upgrade
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python search_ninjas_rank.py train
```

Training reads `cran.all.1400`, `cran.qry`, and `cranqrel`. It creates the
`search_ninjas_artifacts` directory containing the learned models, test-query
rankings, evaluation tables, and `search_ninjas_report.pdf`. The split is
reproducible (seed 42: 135 training, 45 validation, and 45 test queries).

To run the project tests and check the exported evaluation results:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python search_ninjas_verify.py
```

The tests and verifier are included in the repository but are not needed to
run the submitted ranking programs.

## Ranking new queries

The submission includes `search_ninjas_final_model.joblib`, fitted on the
training and validation queries. To rank a different query file:

```bash
.venv/bin/python search_ninjas_rank.py predict --queries new_queries.qry
```

Query files follow the Cranfield convention:

```text
.I 001
.W
aerodynamic wing pressure
.I 002
.W
boundary layer viscosity
```

The command creates `search_ninjas_artifacts/predictions/` with zone and SVM
rankings. Each method has a `_seqid.txt` file (sequential query numbers) and a
`_labels.txt` file (the identifiers appearing after `.I`). By default, the
top 100 documents are returned per query. Use `--top-k` to change the cutoff,
or `--output` to select another directory. No relevance judgments are needed
for prediction.

Every result line follows `query-id Q0 document-id rank score search_ninjas`.
The score column contains the model's ranking score. The program resolves
equal scores by increasing numeric document ID when writing its result files.

## Methods

**Method A - weighted zones.** Each field (title, author, bibliography, body)
receives the fraction of distinct query terms present in it. The four
nonnegative field weights sum to one and are selected by an exhaustive
0.1-step grid search on training F1@10. This is a per-term Boolean extension
of whole-field zone matching; the original any-term OR variant is retained as
the `zone_or` baseline. Scores are calculated from the weighted field values;
the original whole-field OR method is kept for comparison.

**Method B - linear SVM.** The program computes the specified 22 retrieval
features for each of the four fields (88 features in total). A linear
hinge-loss SVM is fitted to query-document relevance labels; its decision
function ranks documents. Regularization is selected on validation F1@10.
The body-only BM25 and equally weighted zone methods provide additional
baselines.

All methods share lowercase alphanumeric tokenization and English stopword
removal. Codes 1-4 in `cranqrel` are treated as relevant; code -1 and
unjudged pairs are treated as nonrelevant. The evaluation reports macro-
averaged precision, recall, and per-query F1 at cutoffs 5, 10, 20, 50, and
100. Details, measured results, and limitations are in the PDF report.

The SVM regularization search and the per-term zone definition were refined
after an earlier examination of test results. Therefore the reported test
split should not be treated as a fully untouched final evaluation.

## Submission

The compact archive `search_ninjas_submission.zip` is generated with:

```bash
.venv/bin/python search_ninjas_package.py
```

It contains the executable programs, required inputs, the final model,
machine-readable test results, and the PDF report. Training produces more
diagnostic files than are necessary for grading; those remain outside the
archive. Only load `.joblib` models from a trusted source.
