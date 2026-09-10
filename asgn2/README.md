# PA2 — Vector Space Model and Ranked Retrieval (group `search_ninjas`)

Ranked retrieval over the Cranfield collection with PyTerrier 1.1.2 / Terrier 5.11,
evaluated against `cranqrel`, with preprocessing, model and parameter tuning.

The write-up required for submission is `search_ninjas_report.pdf`; it is generated from
the experiment output rather than written by hand, so its numbers cannot drift from the
runs. This file is the operational README: how to run things, and what a grader testing
further queries needs to know.

## Running it

```sh
./search_ninjas_setup.sh                                  # venv, JDK, unpack cran.tar.gz
.venv/bin/python search_ninjas_experiments.py             # all four experiment stages
.venv/bin/python search_ninjas_search.py --evaluate       # the tuned run + its scores
.venv/bin/python search_ninjas_report.py                  # the submission PDF
```

Terrier is a Java engine. If the machine has no system JDK the setup script downloads
Temurin 17 into `.venv/jdk`, and `search_ninjas_env.py` points `JAVA_HOME` at it —
nothing is installed outside this directory. Every module that imports `pyterrier`
imports `search_ninjas_env` first, because pyjnius resolves the JVM at import time.

Stage 1 rebuilds 27 indices and takes a few minutes; the rest is seconds. Individual
stages: `--stage 3`, repeatable.

## Running further test queries

This is the part that matters for the unseen-query evaluation.

```sh
.venv/bin/python search_ninjas_search.py --queries your_queries.qry --top-k 100
```

The query file must be in `cran.qry` format (`.I <id>` / `.W` / text). Two run files are
written, in the six-column TREC format `qid Q0 docno rank score search_ninjas`:

| File | Query ids |
| --- | --- |
| `search_ninjas_results.txt` | the query file's own `.I` labels |
| `search_ninjas_results_seqid.txt` | position in the file, `1..N` |

**Both are written because the two numberings disagree for `cran.qry` itself**, and which
one is correct depends on the judgements the run is scored against. `cranqrel` numbers
the 225 queries 1..225 by their *position*, but `cran.qry`'s own `.I` labels are
non-contiguous — 001, 002, 004, 008, … 365. Scoring a label-keyed run against `cranqrel`
silently evaluates nearly every query against the wrong judgements, and the resulting MAP
looks plausible rather than obviously broken. Use `_seqid` against `cranqrel`; use the
other if your judgements carry the query file's own ids.

Query text goes through the same case-folding and non-alphanumeric stripping as the
indexed text before it reaches Terrier's parser. That is not cosmetic: Terrier reads
characters like `/` and `.` as query operators, and every Cranfield question ends in a
period.

## Configuration

`search_ninjas_search.py` takes its entire configuration from
`search_ninjas_experiment_results/best_configuration.json`, which stage 4 writes. The
tuning result is data, not a second copy of the numbers in the code — re-run the
experiments and the search script follows. If the file is absent it falls back to the
defaults in `DEFAULT_CONFIG`.

## Method

**Preprocessing** repeats PA1's steps as Terrier's term pipeline: case folding and
non-alphanumeric stripping (English tokeniser), stopword removal against PA1's
`stopwords.txt`, Porter stemming. All three are switchable — `--fields`, `--stemmer`,
`--stopwords` on `search_ninjas_index.py` — so stage 1 can measure each one.

**Experiments** run in four narrowing stages: preprocessing → weighting model →
parameters → pseudo-relevance feedback. Stages 3 and 4 tune on the **odd-numbered**
queries and report on the **even-numbered** ones as well as on the full set. The
interleaved split rather than a contiguous one is deliberate: the Cranfield queries are
grouped by subject, so cutting the list in half would tune on a different subject mix
than it reports on.

**Metrics** come from `ir_measures`: MAP (used for every selection), nDCG@10, nDCG@20,
P@5, P@10, recall@100, RR. Comparisons against a baseline carry a paired sign test.

Cleverdon's relevance codes run 1 (complete answer) to 4 (minimum interest) — smaller is
better, the reverse of what evaluation tools assume — so `read_qrels` inverts them to
gains 4..1. The 225 rows coded `-1` are unjudged markers and are dropped.

## What the experiments found

| Stage | Finding | Held-out MAP effect |
| --- | --- | --- |
| 1 | Stopword removal is the largest single effect in the study | ≈ +0.11 |
| 1 | Porter stemming, and it beats weak Porter | ≈ +0.03 |
| 1 | Field choice (title+abstract vs abstract vs everything) | ≈ ±0.01 |
| 2 | Model choice among IDF-weighted models is narrow; no-IDF `Tf` collapses | 0.19 → 0.33 |
| 3 | Parameter tuning is **not** significant (p ≈ 0.19) | ≈ +0.005 |
| 4 | Bo1 query expansion **is** significant (p ≈ 0.006) | ≈ +0.023 |

The headline: preprocessing and query expansion carry this collection, parameter tuning
does not. The BM25 response surface is nearly flat — every setting from k₁ 0.8–4.0 and
b 0.3–1.0 lands within about 0.01 MAP. Expansion is also the only change with a real
cost, roughly tripling search time because it retrieves twice.

Full tables are in `search_ninjas_experiment_results/*.csv` and in the report.

## Collection quirks

- A record is written `.T .A .B .W`, so an `.A`/`.B` tag arriving after `.W` has opened is
  stray text inside the abstract, not a new field. Document 240 contains exactly this;
  without the rule it loses ~15 lines of its abstract.
- Documents **471** and **995** are completely empty records. Terrier warns "Indexed 2
  empty documents". Document 995 is judged relevant for query 125, so it is an
  unreachable relevant document that caps achievable recall — not a bug to chase.
- Docids are contiguous 1..1400, body text is already lowercase, and the title is usually
  repeated as the first lines of the abstract, which is why indexing the abstract alone
  scores close to indexing title plus abstract.
- `search_ninjas_stopwords.txt` (carried over from PA1) is UTF-8 with CRLF endings and
  holds four mojibake entries; each line goes through the same non-alphanumeric split the
  tokens do, which is what keeps them matching.
