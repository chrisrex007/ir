# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

IR Programming Assignment 1 (`PA1.pdf`): preprocess the Cranfield collection, build an
inverted index, and answer two-term Boolean queries. Python 3, standard library only.

## Commands

```sh
python3 search_ninjas_preprocess.py                            # cran.all.1400  -> search_ninjas_processed.all
python3 search_ninjas_index.py                                 # processed file -> search_ninjas_cran.index
python3 search_ninjas_search.py "aerodynamics AND slipstream"   # index         -> search_ninjas_results.txt
python3 search_ninjas_search.py --queries search_ninjas_queries.txt     # batch of queries
```

The three stages are sequential: changing preprocessing invalidates the index, which
invalidates the results, so re-run all three. The default file names need no
arguments; `--input` and `--output` override them. The whole pipeline runs in well
under a second.

There is no test suite. Verify changes by cross-checking the programs against each
other — the reliable method is to rebuild the term-to-docid mapping directly from
`search_ninjas_processed.all` and compare it against `search_ninjas_search.search`, and to compare
`search_ninjas_search.lookup` (binary search) against a plain full scan of the index file for
every term in the vocabulary. `intersect`/`union` can be checked against Python set
operations on the same postings lists.

## Assignment constraints that shape the code

- **No IR libraries.** Everything is written from scratch; `search_ninjas_porter.py` implements
  Porter (1980) from the algorithm definition. Only standard string/regex handling is
  allowed. If a reference stemmer is wanted for verification, keep it out of the
  submitted programs.
- **Every program and output file is prefixed with the group name `search_ninjas`.**
  The prefix is fixed: there is deliberately no `--group` flag, and the default paths
  are module constants (`PROCESSED_FILE`, `INDEX_FILE`, `RESULTS_FILE`). Renaming the
  group means editing those constants, the module names and the imports together.
- **Output formats are specified by the assignment** and graders parse them. Do not
  change them casually: `search_ninjas_processed.all` uses `.I <docid>` / `.S` / tokens, and
  `search_ninjas_cran.index` begins with a `<vocabulary size>, <max docid>` header followed by
  `term d1,d2,d3` lines sorted lexicographically.

## Architecture

`search_ninjas_preprocess.py` holds the four pipeline stages as four separate functions
(required by the assignment), each mapping a `{docid: [token]}` dict to another:
`tokenize` → `normalize` → `remove_stopwords` → `stem_tokens`.

Order matters: stopword removal must precede stemming, because `stopwords.txt` is in
surface form. Stemming last also guarantees the index and the queries are reduced by
the same function — `search_ninjas_search.prepare_term` deliberately calls
`search_ninjas_preprocess.normalize` and `search_ninjas_porter.stem` rather than reimplementing them,
so query terms and index terms cannot drift apart.

`search_ninjas_index.py` relies on documents being visited in ascending docid order, so each
postings list is built already sorted and a duplicate can only be the docid just
appended — there is no later sort or de-duplication pass to keep in sync.

`search_ninjas_search.py` exploits both orderings in the index file: terms are sorted, so
`lookup` binary searches the file's byte offsets instead of scanning it, and postings
are sorted, so `intersect`/`union` are linear merges. The binary search is the subtle
part — a seek lands mid-line, so the remainder of that line is skipped before a term
is read, and the byte range must be maintained so a skipped line stays reachable. The
invariant is that every line starting below `low` sorts before the target; `high` may
only ever be pulled down to `middle`, never to a position past a line that has not
been compared. Getting this wrong silently loses terms whose line happens to begin at
a probed offset, and the failure is invisible without a full-vocabulary comparison
against a linear scan.

## Collection quirks

`cran.all.1400` is not uniformly tagged. Document 240 has stray `.A`/`.B` lines inside
its abstract, and documents 576 and 578 each have a second `.W` section. The parser is
a state machine keyed on the most recent field tag, with one extra rule: a record is
written `.T .A .B .W`, so an `.A`/`.B` tag arriving after `.W` has opened is stray text
inside the abstract and does not switch fields. Without that rule document 240 silently
loses the whole remainder of its abstract (~15 lines) to "author" mode, which is enough
to fail sample query 2 (`dynamics OR effects`) by one document — the kind of error that
shows up only against the published expected counts.

Docids are contiguous 1..1400. The body text is already lowercase, and in most
documents the title is repeated as the first lines of the abstract — harmless for a
Boolean index, which ignores term frequency.

`stopwords.txt` is UTF-8 with CRLF endings and holds four mojibake entries (`herse”`,
`himse”`, …); `load_stopwords` runs the list through the text normalizer, which is
what keeps them from breaking the comparison.

## Current output

1400 documents, 247,422 raw tokens, 136,043 after stopword removal, 4,626 distinct
stems, 80,477 postings.

All 12 sample queries in `sample_queries.md` reproduce exactly — every `AND` docid list
matches docid for docid and every `OR` count matches. `search_ninjas_queries.txt` holds them in
both `AND` and `OR` form, so `python3 search_ninjas_search.py --queries search_ninjas_queries.txt`
regenerates the evidence; graders will run further queries later, so re-check this after
any preprocessing change.

`README.md` carries the methodology write-up required for submission, including a "How
queries are processed" section written for whoever tests the additional queries. Update
its statistics table, its lookup-cost figures and that section whenever the pipeline or
the query handling changes.
