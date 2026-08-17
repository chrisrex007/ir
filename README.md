# IR Programming Assignment 1 — Preprocessing, Indexing, Boolean Search

Cranfield collection (1400 documents), preprocessed and indexed from scratch, with
a Boolean retrieval program for two-term `AND` / `OR` queries. Python 3, standard
library only — no information retrieval library is used, and the Porter stemmer is
implemented directly from the published algorithm.

## Files

| File | Description |
| --- | --- |
| `group_preprocess.py` | Tokenization, normalization, stopword removal, stemming |
| `group_porter.py` | Porter (1980) stemming algorithm |
| `group_index.py` | Builds the inverted index |
| `group_search.py` | Boolean search over the index file |
| `group_processed.all` | Preprocessed collection (output) |
| `group_cran.index` | Inverted index (output) |
| `group_results.txt` | Query results (output) |
| `group_queries.txt` | Sample test queries |
| `cran.all.1400` | Raw Cranfield collection (input) |
| `stopwords.txt` | Stopword list (input) |

`group` is the file-name prefix; every program accepts `--group <name>` to change it.

## Running

```sh
python3 group_preprocess.py                          # -> group_processed.all
python3 group_index.py                               # -> group_cran.index
python3 group_search.py "aerodynamics AND slipstream" # -> group_results.txt
python3 group_search.py --queries group_queries.txt   # a batch of queries
```

Each program takes `--input`, `--output` and `--group` if the defaults need to
change; `python3 <program> --help` lists them.

## Methodology

### Preprocessing

The four stages are four separate functions in `group_preprocess.py`, applied to
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

The output, `group_processed.all`, holds one document per record:

```
.I 1
.S
experiment investig aerodynam wing slipstream ...
```

### Indexing

`group_index.py` reads the processed file and inverts it. Documents are visited in
ascending docid order, so each term's postings list is built already sorted and a
duplicate can only be the docid just appended — no sorting or de-duplication pass
is needed afterwards. Terms are then written in lexicographical order to
`group_cran.index`:

```
4625, 1400
aerodynam 1,7,10,...
```

The first line is the vocabulary size and the largest indexed docid. Every
following line is one term, a space, and its postings list of ascending docids
separated by commas.

### Boolean search

`group_search.py` answers `<word1> AND|OR <word2>`. The query words go through the
same `normalize` and Porter `stem` functions as the collection, so `aerodynamics`
becomes `aerodynam` and matches the index.

Two properties of the index file are used to keep the search cheap:

- **Term lookup is a binary search over the file's byte offsets.** The index is
  sorted lexicographically, so the file is halved by seeking, rather than scanned.
  A seek lands in the middle of a line, so the rest of that line is skipped before
  a term is read; the byte range is maintained such that every line beginning below
  its lower bound sorts before the target, which keeps the skipped lines reachable.
  Measured over all 4625 terms, the worst lookup costs 20 seeks and reads 42 lines
  out of 4625 — logarithmic rather than linear in the vocabulary.
- **The two postings lists are combined by a linear merge**, walking both lists
  once with two indices: `intersect` for `AND`, `union` for `OR`. This costs
  O(len(p1) + len(p2)) and needs no hashing or sorting, because the postings are
  already in ascending docid order.

Results are written to `group_results.txt`, one block per query:

```
Query: aerodynamics AND slipstream
Stemmed query: aerodynam AND slipstream
Matched documents: 5
1,453,1064,1089,1164
```

## Collection statistics

| Quantity | Value |
| --- | --- |
| Documents | 1400 |
| Raw tokens | 247,273 |
| After normalization | 242,923 |
| After stopword removal | 135,961 |
| Vocabulary (distinct stems) | 4,625 |
| Postings | 80,445 |
| Maximum docid | 1400 |

## Notes on the collection

Three documents deviate from the regular `.T .A .B .W` tag order, and the parser
handles them by attributing text to whichever field tag most recently opened:

- **Document 240** has stray `.A` and `.B` lines inside its abstract, so the two
  abstract lines that follow them are treated as author/affiliation text and are
  not indexed.
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
