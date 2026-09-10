"""Build a Terrier index over the Cranfield collection under a chosen preprocessing variant."""

import argparse
import os
import re
import shutil
import time

import search_ninjas_env  # noqa: F401  -- must precede pyterrier, sets JAVA_HOME
import pyterrier as pt

import search_ninjas_parse as parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_ROOT = os.path.join(BASE_DIR, "search_ninjas_indices")
STOPWORD_FILE = os.path.join(BASE_DIR, "search_ninjas_stopwords.txt")

# The abstract repeats the title in most records; author and bib are names and references.
FIELD_SETS = {
    "text": ["text"],
    "title_text": ["title", "text"],
    "all": ["title", "text", "author", "bib"],
}

STEMMERS = ("none", "porter", "weakporter")
STOPWORD_SETS = ("assignment", "terrier", "none")

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def load_stopwords(path=STOPWORD_FILE):
    """Read the assignment stopword list, split the same way the document tokens are."""
    words = set()
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            for piece in _NON_ALNUM.split(line.strip().lower()):
                if piece:
                    words.add(piece)
    return sorted(words)


def variant_name(fields, stemmer, stopwords):
    return f"{fields}__{stemmer}__{stopwords}"


def index_path(fields, stemmer, stopwords):
    return os.path.join(INDEX_ROOT, variant_name(fields, stemmer, stopwords))


def build_index(fields="all", stemmer="porter", stopwords="assignment", overwrite=False, verbose=False):
    """Build or reuse one preprocessing variant; returns (index, build seconds)."""
    path = index_path(fields, stemmer, stopwords)
    if os.path.exists(path) and not overwrite:
        return pt.IndexFactory.of(path), 0.0
    if os.path.exists(path):
        shutil.rmtree(path)

    if stopwords == "assignment":
        stopword_arg = load_stopwords()
    elif stopwords == "terrier":
        stopword_arg = "terrier"
    else:
        stopword_arg = None

    documents = parse.read_documents()
    indexer = pt.terrier.IterDictIndexer(
        path,
        text_attrs=FIELD_SETS[fields],
        meta={"docno": 8},
        stemmer=stemmer,
        stopwords=stopword_arg,
        tokeniser="english",
        verbose=verbose,
    )

    start = time.perf_counter()
    index_ref = indexer.index(documents)
    elapsed = time.perf_counter() - start
    return pt.IndexFactory.of(index_ref), elapsed


def describe(index):
    """Return the collection statistics Terrier keeps for an index."""
    stats = index.getCollectionStatistics()
    return {
        "documents": stats.getNumberOfDocuments(),
        "terms": stats.getNumberOfUniqueTerms(),
        "postings": stats.getNumberOfPointers(),
        "tokens": stats.getNumberOfTokens(),
        "avg_doc_length": round(stats.getAverageDocumentLength(), 2),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", default="all", choices=sorted(FIELD_SETS))
    ap.add_argument("--stemmer", default="porter", choices=sorted(STEMMERS))
    ap.add_argument("--stopwords", default="assignment", choices=STOPWORD_SETS)
    ap.add_argument("--overwrite", action="store_true", help="rebuild even if the index exists")
    args = ap.parse_args()

    index, elapsed = build_index(args.fields, args.stemmer, args.stopwords, args.overwrite, verbose=True)
    print(f"variant       : {variant_name(args.fields, args.stemmer, args.stopwords)}")
    print(f"path          : {index_path(args.fields, args.stemmer, args.stopwords)}")
    print(f"indexing time : {elapsed:.2f} s" if elapsed else "indexing time : (reused existing index)")
    for key, value in describe(index).items():
        print(f"{key:14}: {value}")


if __name__ == "__main__":
    main()
