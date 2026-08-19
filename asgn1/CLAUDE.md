# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

IR Programming Assignment 1 (`PA1.pdf`): preprocess the Cranfield collection, build an
inverted index, and answer two-term Boolean queries. Python 3, standard library only.

## Commands

```sh
python3 group_preprocess.py                            # cran.all.1400  -> group_processed.all
python3 group_index.py                                 # processed file -> group_cran.index
python3 group_search.py "aerodynamics AND slipstream"   # index         -> group_results.txt
python3 group_search.py --queries group_queries.txt     # batch of queries
```

The three stages are sequential: changing preprocessing invalidates the index, which
invalidates the results, so re-run all three. Every program takes `--group`, `--input`
and `--output`; the whole pipeline runs in well under a second.

There is no test suite. Verify changes by cross-checking the programs against each
other — the reliable method is to rebuild the term-to-docid mapping directly from
`group_processed.all` and compare it against `group_search.search`, and to compare
`group_search.lookup` (binary search) against a plain full scan of the index file for
every term in the vocabulary. `intersect`/`union` can be checked against Python set
operations on the same postings lists.

## Assignment constraints that shape the code

- **No IR libraries.** Everything is written from scratch; `group_porter.py` implements
  Porter (1980) from the algorithm definition. Only standard string/regex handling is
  allowed. If a reference stemmer is wanted for verification, keep it out of the
  submitted programs.
- **Every program and output file is prefixed with the group name.** It is currently
  the placeholder `group`; `--group <name>` changes the prefix on both inputs and
  outputs, so renaming means re-running the pipeline rather than editing filenames.
- **Output formats are specified by the assignment** and graders parse them. Do not
  change them casually: `group_processed.all` uses `.I <docid>` / `.S` / tokens, and
  `group_cran.index` begins with a `<vocabulary size>, <max docid>` header followed by
  `term d1,d2,d3` lines sorted lexicographically.

## Architecture

`group_preprocess.py` holds the four pipeline stages as four separate functions
(required by the assignment), each mapping a `{docid: [token]}` dict to another:
`tokenize` → `normalize` → `remove_stopwords` → `stem_tokens`.

Order matters: stopword removal must precede stemming, because `stopwords.txt` is in
surface form. Stemming last also guarantees the index and the queries are reduced by
the same function — `group_search.prepare_term` deliberately calls
`group_preprocess.normalize` and `group_porter.stem` rather than reimplementing them,
so query terms and index terms cannot drift apart.

`group_index.py` relies on documents being visited in ascending docid order, so each
postings list is built already sorted and a duplicate can only be the docid just
appended — there is no later sort or de-duplication pass to keep in sync.

`group_search.py` exploits both orderings in the index file: terms are sorted, so
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
its abstract (so two abstract lines are read as author text and skipped), and
documents 576 and 578 each have a second `.W` section. The parser is a state machine
keyed on the most recent field tag, which absorbs all three cases; a parser that
assumes exactly one `.T .A .B .W` group per document will not.

Docids are contiguous 1..1400. The body text is already lowercase, and in most
documents the title is repeated as the first lines of the abstract — harmless for a
Boolean index, which ignores term frequency.

`stopwords.txt` is UTF-8 with CRLF endings and holds four mojibake entries (`herse”`,
`himse”`, …); `load_stopwords` runs the list through the text normalizer, which is
what keeps them from breaking the comparison.

## Current output

1400 documents, 247,273 raw tokens, 135,961 after stopword removal, 4,625 distinct
stems, 80,445 postings. `README.md` carries the full methodology write-up required for
submission; update its statistics table when the preprocessing changes.
