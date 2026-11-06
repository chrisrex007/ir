# Assignment 3: Learning to Rank

Standalone implementation of the uploaded PA3 brief. Python 3.12; no Java or prior
assignment modules required. Inputs are the supplied `PA3.pdf` and Cranfield files.

## Run

```sh
git clone https://github.com/chrisrex007/ir.git
cd ir/asgn3
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python search_ninjas_rank.py train
```

Training runs both methods, evaluation, baselines, and report generation.
The report always uses `search_ninjas` and the five members' names and student IDs
recorded in `search_ninjas_group.py`.
Use `--output PATH` to keep multiple experimental runs.

```sh
.venv/bin/python search_ninjas_rank.py predict --queries your_queries.qry
```

Prediction defaults to the separately refitted final model, requires no judgments, and scores all 1,400
documents for each query. Query files use `.I <label>` followed by `.W` and text.
`--top-k` defaults to 100. Ties use ascending numeric document ID. Empty/OOV queries
are retained with deterministic rankings, although they have no lexical evidence.
Only load trusted joblib files.

## Methods and experimental plan

1. Parse title, author, bibliography, body; lowercase alphanumeric tokenization,
   scikit-learn English stopwords, no stemming. Preserve empty documents and stray
   tags inside abstracts.
2. Split queries 135/45/45 into training/validation/test using seed 42. Qrels use
   **query positions**, not the non-contiguous original `.I` labels. Codes 1–4 are
   relevant; -1, code 5, and unjudged pairs are treated as nonrelevant in both
   training and evaluation. There are 1,612 positive judgments among 1,837 rows.
   This follows the [ir_datasets convention](https://github.com/allenai/ir_datasets/blob/master/ir_datasets/datasets/cranfield.py),
   which calls -1 “References of no interest.”
3. **A:** Boolean OR match per field. Search the 286 nonnegative weight tuples
   summing to one at increments of 0.1. Select using training macro F1@10.
4. **B:** Compute the brief's 22 features per field (88 total). Train a pointwise
   hinge-loss linear SVM (SGDClassifier, averaged coefficients) with balanced classes on all training query-document pairs. Scale
   using training pairs only; select L2 alpha from .00001/.0001/.001/.01/.1/1/10/100 on validation F1@10.
   Retain that model without refitting. Rank by signed decision score.
5. Evaluate on held-out queries at k=5,10,20,50,100. Report macro precision,
   recall and per-query F1 averages. Compare equal-zone and body BM25 baselines.

The feature definitions, formulas, assumptions, coefficients, observed results,
limitations, and reproduction instructions appear in the generated PDF. Corpus
statistics use all documents, but relevance labels are isolated by query split.
This evaluates new queries against a fixed document collection.

## Outputs

`search_ninjas_artifacts/` contains the model, split/metadata JSON (including input
hashes), tuning CSVs, all 88 SVM weights, aggregate/per-query metrics, held-out TREC
runs, and `search_ninjas_report.pdf`. These generated artifacts are tracked by Git.

Runs ending `_seqid.txt` use positional IDs for Cranfield judgments. Runs ending
`_labels.txt` preserve the query file labels. Each line has six columns:
`qid Q0 docno rank score search_ninjas`. Training writes only held-out test query
runs; `predict` writes runs for every query in the supplied file.

Regenerate the PDF from existing results with:

```sh
.venv/bin/python search_ninjas_report.py
```

The implementation does not submit or upload anything to Moodle.

## Revised experiments and final model

This revision follows inspection of the original test results. Alpha selection
still uses validation F1@10 only, but the revised test results are not a fresh,
previously untouched evaluation. No statistical superiority is claimed.

`zone_bm25_tiebreak` is an exploratory hybrid: zone score first (rounded to 12
decimals), body BM25 within ties, then document ID. It keeps the pure method's
learned weights. Its TREC scores encode descending ordinal rank; pure `zone`
remains the required Method A. Diagnostics quantify tied weight optima and scores.

`search_ninjas_model.joblib` is the evaluation model trained on 135 queries.
`search_ninjas_final_model.joblib` refits the scaler and SVM on 180 train+validation
queries at the selected alpha, and relearns zone weights on those queries.
The 45 test queries remain excluded. Reported metrics belong only to the evaluation
model. Both bundles have separate metadata describing their fitting queries.

To reproduce evaluation-model predictions, supply:

```sh
.venv/bin/python search_ninjas_rank.py predict --queries your_queries.qry --model search_ninjas_artifacts/search_ninjas_model.joblib --output evaluation_predictions
```

After training, independently verify all exported test runs and build a submission:

```sh
.venv/bin/python search_ninjas_verify.py
.venv/bin/python search_ninjas_package.py
```

The archive is `search_ninjas_submission.zip` in this directory. It includes source,
tests, pinned requirements, inputs, PDF, ranking files, and both models. It excludes
the virtual environment and includes a SHA-256 manifest. No submission is uploaded
automatically. On Windows, use `.venv\Scripts\python.exe` instead of `.venv/bin/python`.
