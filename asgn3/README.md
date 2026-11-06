# Assignment 3: Learning to Rank

Standalone implementation of the uploaded PA3 brief. Python 3.12; no Java or prior
assignment modules required. Inputs are the supplied `PA3.pdf` and Cranfield files.

## Run

```sh
cd /workspace/ir/asgn3
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

Prediction loads the saved model, requires no judgments, and scores all 1,400
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
   relevant, -1 is ignored, and unjudged pairs are treated as nonrelevant.
3. **A:** Boolean OR match per field. Search the 286 nonnegative weight tuples
   summing to one at increments of 0.1. Select using training macro F1@10.
4. **B:** Compute the brief's 22 features per field (88 total). Train a pointwise
   hinge-loss linear SVM (SGDClassifier, averaged coefficients) with balanced classes on all training query-document pairs. Scale
   using training pairs only; select L2 alpha from .00001/.0001/.001/.01 on validation F1@10.
   Retain that model without refitting. Rank by signed decision score.
5. Evaluate once on held-out queries at k=5,10,20,50,100. Report macro precision,
   recall and per-query F1 averages. Compare equal-zone and body BM25 baselines.

The feature definitions, formulas, assumptions, coefficients, observed results,
limitations, and reproduction instructions appear in the generated PDF. Corpus
statistics use all documents, but relevance labels are isolated by query split.
This evaluates new queries against a fixed document collection.

## Outputs

`search_ninjas_artifacts/` contains the model, split/metadata JSON (including input
hashes), tuning CSVs, all 88 SVM weights, aggregate/per-query metrics, held-out TREC
runs, and `search_ninjas_report.pdf`. Generated artifacts are ignored by Git; include
them explicitly in the submission archive alongside the source programs.

Runs ending `_seqid.txt` use positional IDs for Cranfield judgments. Runs ending
`_labels.txt` preserve the query file labels. Each line has six columns:
`qid Q0 docno rank score search_ninjas`. Training writes only held-out test query
runs; `predict` writes runs for every query in the supplied file.

Regenerate the PDF from existing results with:

```sh
.venv/bin/python search_ninjas_report.py
```

The implementation does not submit or upload anything to Moodle.
