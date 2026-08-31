# IR Programming Assignment 1 — Preprocessing, Indexing, Boolean Search

Cranfield collection (1400 documents), preprocessed and indexed from scratch, with
a Boolean retrieval program for two-term `AND` / `OR` queries. Python 3, standard
library only — no information retrieval library is used, and the Porter stemmer is
implemented directly from the published algorithm.

**Verified against all 12 sample queries: every `AND` docid list and every `OR`
count matches exactly.** See [Verification](#verification).

## Files

| File | Description |
| --- | --- |
| `search_ninjas_preprocess.py` | Tokenization, normalization, stopword removal, stemming |
| `search_ninjas_porter.py` | Porter (1980) stemming algorithm |
| `search_ninjas_index.py` | Builds the inverted index |
| `search_ninjas_search.py` | Boolean search over the index file |
| `search_ninjas_processed.all` | Preprocessed collection (output) |
| `search_ninjas_cran.index` | Inverted index (output) |
| `search_ninjas_results.txt` | Query results (output) |
| `search_ninjas_queries.txt` | The 12 sample queries, in both `AND` and `OR` form |
| `cran.all.1400` | Raw Cranfield collection (input) |
| `stopwords.txt` | Stopword list (input) |

Every program and output file carries the group name `search_ninjas` as its prefix,
as the assignment requires.

## Running

```sh
python3 search_ninjas_preprocess.py                            # -> search_ninjas_processed.all
python3 search_ninjas_index.py                                 # -> search_ninjas_cran.index
python3 search_ninjas_search.py "aeroelastic AND aircraft"     # -> search_ninjas_results.txt
python3 search_ninjas_search.py --queries search_ninjas_queries.txt    # a batch of queries
```

The file names above are the defaults and need no arguments. Each program also
accepts `--input` and `--output` should a file need to be read from or written to
somewhere else; `python3 <program> --help` lists them. The full pipeline runs in
under a second.

---

## How queries are processed

This section is the one to read when testing the programs against additional
queries.

### Running a query

A single query is passed as one quoted command-line argument:

```sh
python3 search_ninjas_search.py "aeroelastic AND aircraft"
```

A batch of queries is passed as a file holding **one query per line**, which is the
easier route for a set of test queries:

```sh
python3 search_ninjas_search.py --queries your_queries.txt
```

Both write to `search_ninjas_results.txt`, or to whatever `--output <file>` names. The
program exits with status 0 when at least one query was answered, and 1 when none
could be parsed.

### Accepted query form

A query is exactly three whitespace-separated fields:

```
<word1> AND <word2>
<word1> OR  <word2>
```

- The operator may be written in any case — `AND`, `and`, `Or` are all accepted.
- Only the two connectives `AND` and `OR` are supported, as the assignment
  specifies; there is no `NOT`, no parentheses, and no third term.
- A malformed query (wrong number of fields, or an unknown operator) is reported on
  stderr and skipped, and the remaining queries in a batch are still answered.

### What happens to each query word

Each of the two query words is put through **exactly the same normalization and
stemming as the collection was**, in the same order:

1. **Case-folding** — the word is lowercased.
2. **Possessive removal** — a trailing `'s` is dropped, so `earth's` becomes `earth`.
3. **Splitting on non-alphanumeric characters** — every character that is not a
   letter or a digit is treated as a separator, so `re-entry` yields `re` and
   `entry`, and `/slip/` yields `slip`. If a query word splits into more than one
   piece, the first piece is used as the query term.
4. **Porter stemming** — the result is reduced by the same Porter implementation
   used to build the index, so `aeroelastic` → `aeroelast`, `viscosity` → `viscos`,
   `oscillatory` → `oscillatori`, `nozzle` → `nozzl`.

This is not a reimplementation: `search_ninjas_search.prepare_term` calls
`search_ninjas_preprocess.normalize` and `search_ninjas_porter.stem` directly, so a query term and
an index term can never be reduced by different code.

**Stopword removal is deliberately not applied to query words.** It does not need
to be: stopwords were removed when the index was built, so a stopword query term
simply finds no postings. `the AND flow` therefore returns nothing and
`the OR flow` returns exactly the documents containing `flow`, which is the correct
Boolean answer either way.

A query term that is not in the vocabulary is treated as having an empty postings
list — the `AND` result is empty, and the `OR` result is the other term's postings.

### How the query is answered

The two stemmed terms are looked up in `search_ninjas_cran.index` by **binary search over
the file's byte offsets**, and their postings lists are combined by a **linear
merge** — `intersect` for `AND`, `union` for `OR`. Both are described under
[Boolean search](#boolean-search) below.

### Output format

`search_ninjas_results.txt` holds one block per query, blocks separated by a blank line:

```
Query: aeroelastic AND aircraft
Stemmed query: aeroelast AND aircraft
Matched documents: 5
12,14,78,184,202
```

The four lines are the query as it was given, the two terms after normalization and
stemming, the number of matching documents, and the matching docids in ascending
order, separated by commas. The docid line is empty when nothing matched. The
`Stemmed query` line is there to make a mismatch easy to diagnose: if a result
looks wrong, it shows immediately whether the cause was the stemmer or the merge.

---

## Methodology

### Preprocessing

The four stages are four separate functions in `search_ninjas_preprocess.py`, applied to
the whole collection in this order:

1. **`tokenize(path)`** reads `cran.all` and splits it into raw token strings.
   A document starts at a `.I <docid>` line, and only the title (`.T`) and abstract
   (`.W`) fields are kept — the author (`.A`) and affiliation (`.B`) fields are
   ignored, as the assignment specifies. Tokens are the whitespace-separated
   pieces of those fields.
2. **`normalize(documents)`** case-folds each token, drops a possessive `'s`, and
   splits on every character that is not a letter or a digit. This discards
   punctuation, and also breaks hyphenated compounds — `boundary-layer` becomes
   `boundary` and `layer`, so either half retrieves the compound. Tokens that
   carry no alphanumeric content at all are dropped.
3. **`remove_stopwords(documents, stopwords)`** drops tokens present in
   `stopwords.txt`. The stopword list is itself passed through the same
   normalization, so the two vocabularies are comparable.
4. **`stem_tokens(documents)`** replaces each token with its Porter stem, caching
   stems so that a repeated word is stemmed once.

Normalization and stopword removal both run *before* stemming: the stopword list
is written in surface form, so `the` has to be matched and removed before Porter
would rewrite it. Stemming last also means the index and the queries are reduced
by exactly the same function.

The output, `search_ninjas_processed.all`, holds one document per record:

```
.I 1
.S
experiment investig aerodynam wing slipstream ...
```

### Indexing

`search_ninjas_index.py` reads the processed file and inverts it. Documents are visited in
ascending docid order, so each term's postings list is built already sorted and a
duplicate can only be the docid just appended — no sorting or de-duplication pass
is needed afterwards. Terms are then written in lexicographical order to
`search_ninjas_cran.index`:

```
4626, 1400
aeroelast 12,14,78,...
```

The first line is the vocabulary size and the largest indexed docid. Every
following line is one term, a space, and its postings list of ascending docids
separated by commas.

### Boolean search

`search_ninjas_search.py` answers `<word1> AND|OR <word2>`, processing the query words as
described under [How queries are processed](#how-queries-are-processed).

Two properties of the index file are used to keep the search cheap:

- **Term lookup is a binary search over the file's byte offsets.** The index is
  sorted lexicographically, so the file is halved by seeking, rather than scanned.
  A seek lands in the middle of a line, so the rest of that line is skipped before
  a term is read; the byte range is maintained such that every line beginning below
  its lower bound sorts before the target, which keeps the skipped lines reachable.
  Measured over all 4626 terms, the worst lookup costs 20 seeks and reads 42 lines
  out of 4626 — logarithmic rather than linear in the vocabulary.
- **The two postings lists are combined by a linear merge**, walking both lists
  once with two indices: `intersect` for `AND`, `union` for `OR`. This costs
  O(len(p1) + len(p2)) and needs no hashing or sorting, because the postings are
  already in ascending docid order.

---

## Verification

`search_ninjas_queries.txt` holds all 12 sample queries in both their `AND` and `OR` form,
and `search_ninjas_results.txt` holds the answers produced by these programs. Every `AND`
docid list matches the published expected result exactly, docid for docid, and
every `OR` count matches:

| # | Query | Expected \|AND\| | Ours | Expected \|OR\| | Ours |
| --- | --- | --- | --- | --- | --- |
| 1 | `aeroelastic` / `aircraft` | 5 | 5 | 84 | 84 |
| 2 | `dynamics` / `effects` | 34 | 34 | 586 | 586 |
| 3 | `hypersonic` / `wake` | 5 | 5 | 211 | 211 |
| 4 | `flutter` / `steady` | 11 | 11 | 151 | 151 |
| 5 | `viscosity` / `reynolds` | 23 | 23 | 239 | 239 |
| 6 | `heat` / `stagnation` | 80 | 80 | 360 | 360 |
| 7 | `oscillatory` / `transonic` | 1 | 1 | 83 | 83 |
| 8 | `creep` / `buckling` | 26 | 26 | 136 | 136 |
| 9 | `pressure` / `wing` | 91 | 91 | 687 | 687 |
| 10 | `transonic` / `nozzle` | 3 | 3 | 140 | 140 |
| 11 | `excitation` / `noise` | 5 | 5 | 47 | 47 |
| 12 | `mass` / `flutter` | 6 | 6 | 121 | 121 |

The programs were additionally cross-checked against each other:

- `lookup` (binary search) returns the same postings as a plain linear scan of the
  index file, for all 4626 terms in the vocabulary, and finds nothing for absent
  terms that sort before the first line, after the last line, or between two lines.
- `intersect` and `union` agree with Python set operations over 3000 random term
  pairs.
- The docid sets produced by `search_ninjas_search.py` agree with the term-to-docid mapping
  rebuilt directly from `search_ninjas_processed.all`, which checks the index builder
  independently of the search program.

## Collection statistics

| Quantity | Value |
| --- | --- |
| Documents | 1400 |
| Raw tokens | 247,422 |
| After normalization | 243,068 |
| After stopword removal | 136,043 |
| Vocabulary (distinct stems) | 4,626 |
| Postings | 80,477 |
| Maximum docid | 1400 |

## Notes on the collection

Three documents deviate from the regular `.T .A .B .W` tag order. Because a record
is written as `.T .A .B .W`, an `.A` or `.B` tag that appears once the abstract has
already begun is stray text inside the abstract rather than the start of a new
author field, and the parser treats it that way:

- **Document 240** has stray `.A` and `.B` lines in the middle of its abstract.
  A parser that honours them switches to "author" mode and silently discards the
  whole remainder of the abstract — about fifteen lines. Ignoring them keeps the
  document's text intact, which is what makes docid 240 appear in query 2
  (`dynamics OR effects`).
- **Documents 576 and 578** each contain a second `.W` section, whose text is
  appended to the same abstract.

The body text of the collection is already lowercase, but normalization still
case-folds so that the programs do not depend on that. In most documents the title
is repeated as the opening lines of the abstract; the duplication only affects term
frequencies, which a Boolean index does not use.

`stopwords.txt` is UTF-8 with CRLF line endings and contains four mojibake entries
(`herse”`, `himse”`, `itse”`, `myse”`, originally `herself` and so on). They are
loaded through the same normalization as the text, which reduces them to `herse`,
`himse`, `itse` and `myse` — harmless, since no such tokens occur in the
collection.
