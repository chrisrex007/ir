# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

IR Programming Assignment 2 (`PA2.pdf`): ranked retrieval over the Cranfield collection
using PyTerrier/Terrier, evaluated against `cranqrel`, with parameter and model tuning.
Unlike PA1 — which forbade IR libraries — this one requires an open-source engine, so
nothing here is written from scratch except the collection parsing.

## Environment

Everything lives in `.venv` (gitignored). Terrier is Java, and this machine has no
system JDK, so `search_ninjas_setup.sh` downloads Temurin 17 into `.venv/jdk` and
`search_ninjas_env.py` points `JAVA_HOME` at it. **Every module that imports
`pyterrier` must `import search_ninjas_env` first** — pyjnius resolves the JVM at
import time and fails with `Unable to find javac` otherwise.

```sh
./search_ninjas_setup.sh                             # venv + JDK + untar cran.tar.gz
.venv/bin/python search_ninjas_parse.py              # sanity-check the collection parse
.venv/bin/python search_ninjas_index.py --overwrite  # build one index variant
.venv/bin/python search_ninjas_experiments.py        # all four experiment stages
.venv/bin/python search_ninjas_experiments.py --stage 3   # just one stage
.venv/bin/python search_ninjas_search.py --evaluate  # tuned run -> search_ninjas_results.txt
```

Stage 1 rebuilds 27 indices, so the full experiment driver takes a few minutes; the
other stages are seconds. There is no test suite — correctness is checked by the
collection statistics in `search_ninjas_parse.py`'s `__main__` and by whether the
evaluation numbers stay in the range recorded in `README.md`.

## The two numbering traps

These are the errors that produce plausible-looking but wrong results, so check them
first when a number moves unexpectedly.

1. **Query ids.** `cranqrel` numbers the queries **1..225 by their position** in
   `cran.qry`, but `cran.qry`'s own `.I` labels are non-contiguous (001, 002, 004,
   008, ... 365). Keying a run on the `.I` labels and scoring it against `cranqrel`
   silently scores almost every query against the wrong judgements. `read_queries`
   returns both: `qid` is the position (use this for evaluation), `original_id` is the
   file's label. `search_ninjas_search.py` writes the run twice, once under each.
2. **Relevance grades.** Cleverdon's codes run 1 (complete answer) to 4 (minimum
   interest) — *smaller is better*, the reverse of what every evaluation tool assumes.
   `read_qrels` flips them to gains 4..1. Feeding the raw codes to nDCG inverts the
   ranking of the judgements. Rows coded `-1` (one per query, an unjudged marker) are
   dropped.

## Assignment constraints that shape the code

- **Every program and output file is prefixed with the group name `search_ninjas`.**
  As in PA1 the prefix is fixed — module constants, not a flag.
- **Sparse lexical models only.** No dense or neural retrieval. The weighting models in
  `search_ninjas_experiments.MODELS` are all term-weighting formulas over one inverted
  index; the language-model entries are included for reference and flagged as such in
  the report rather than being submitted as the final system.
- **Indexing and search time must be reported**, so `build_index` returns its elapsed
  time and only times a real build (a reused index reports 0.0), and `timed_run`
  measures retrieval separately.

## Architecture

`search_ninjas_parse.py` is the only place that touches the raw files. Its document
parser is a state machine keyed on the last field tag, carried over from PA1, with the
rule that a record is written `.T .A .B .W`, so an `.A`/`.B` arriving after `.W` has
opened is stray text inside the abstract rather than a new field — without it document
240 loses ~15 lines of its abstract.

`search_ninjas_index.py` exposes preprocessing as three switchable axes (fields,
stemmer, stopword list) and names each index after its variant, so stage 1 can compare
them and the later stages just ask for the winner by name.

`search_ninjas_experiments.py` runs four narrowing stages: preprocessing → model →
parameters → query expansion. Parameters are tuned on **odd** query ids and reported on
**even** ones. The odd/even interleave is deliberate: the Cranfield queries are grouped
by subject, so a contiguous split would tune on a different subject mix than it reports
on. Stage 4 writes `best_configuration.json`, which is the only thing
`search_ninjas_search.py` reads — the tuning result is data, not a second copy of the
numbers in the code.

## Collection quirks

- Documents **471** and **995** are empty records (`.T .A .B .W` with no text at all).
  Terrier warns "Indexed 2 empty documents". Document 995 is judged relevant for query
  125, so it is an unreachable relevant document that caps achievable recall — do not
  chase that missing point.
- Docids are contiguous 1..1400; body text is already lowercase; the title is usually
  repeated as the first lines of the abstract, which is why the `text`-only field set
  scores close to `title_text`.
- `search_ninjas_stopwords.txt` (copied from PA1) is UTF-8 with CRLF endings and holds
  four mojibake entries; `load_stopwords` runs each line through the same
  non-alphanumeric split the tokens get, which is what keeps them matching.
