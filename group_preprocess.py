#!/usr/bin/env python3
"""Preprocess the Cranfield collection: tokenize, normalize, remove stopwords, stem.

Reads the raw collection (cran.all) and writes <group>_processed.all, where each
document is annotated with a .I tag for the docid and a .S tag for its tokens.

Usage:
    python3 group_preprocess.py [--input cran.all.1400] [--stopwords stopwords.txt]
                                [--group group]
"""

import argparse
import re

import group_porter

# Only the title (.T) and abstract (.W) are indexed; authors (.A) and their
# affiliation (.B) are ignored, as required by the assignment.
CONTENT_FIELDS = ("T", "W")

_DOC_TAG = re.compile(r"^\.I\s+(\d+)\s*$")
_FIELD_TAG = re.compile(r"^\.([ITABW])\s*$")

# Normalization splits a raw token wherever a character is not a letter or digit,
# which also breaks hyphenated compounds such as "boundary-layer" into two terms.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# The possessive is dropped first, so that "prandtl's" yields the single term
# "prandtl" rather than "prandtl" plus a stray "s".
_POSSESSIVE = re.compile(r"'s(?![a-z])")


def tokenize(input_path):
    """Read the collection file and split each document into raw token strings.

    Returns a dict mapping docid (int) to its list of whitespace-separated tokens,
    taken from the title and abstract fields only.
    """
    documents = {}
    docid = None
    field = None

    with open(input_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip("\n")

            doc_match = _DOC_TAG.match(line)
            if doc_match:
                docid = int(doc_match.group(1))
                documents[docid] = []
                field = None
                continue

            field_match = _FIELD_TAG.match(line)
            if field_match:
                field = field_match.group(1)
                continue

            if docid is not None and field in CONTENT_FIELDS:
                documents[docid].extend(line.split())

    return documents


def normalize(documents):
    """Case-fold the tokens and strip every non-alphanumeric character.

    Tokens that hold no alphanumeric content at all (bare punctuation) are dropped.
    """
    normalized = {}
    for docid, tokens in documents.items():
        output = []
        for token in tokens:
            token = _POSSESSIVE.sub("", token.lower())
            for piece in _NON_ALNUM.split(token):
                if piece:
                    output.append(piece)
        normalized[docid] = output
    return normalized


def remove_stopwords(documents, stopwords):
    """Drop every token that appears in the stopword list."""
    return {
        docid: [token for token in tokens if token not in stopwords]
        for docid, tokens in documents.items()
    }


def stem_tokens(documents):
    """Reduce every token to its Porter stem, caching repeated words."""
    cache = {}
    stemmed = {}
    for docid, tokens in documents.items():
        output = []
        for token in tokens:
            stem = cache.get(token)
            if stem is None:
                stem = group_porter.stem(token)
                cache[token] = stem
            output.append(stem)
        stemmed[docid] = output
    return stemmed


def load_stopwords(path):
    """Read the stopword list, one word per line, normalized the same way as the text."""
    stopwords = set()
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            for piece in _NON_ALNUM.split(line.strip().lower()):
                if piece:
                    stopwords.add(piece)
    return stopwords


def write_processed(documents, output_path):
    """Write the processed collection with .I (docid) and .S (tokens) annotations."""
    with open(output_path, "w", encoding="utf-8") as handle:
        for docid in sorted(documents):
            handle.write(".I %d\n" % docid)
            handle.write(".S\n")
            handle.write(" ".join(documents[docid]) + "\n")


def preprocess(input_path, stopwords_path, output_path):
    """Run the full pipeline and report what each stage produced."""
    stopwords = load_stopwords(stopwords_path)

    documents = tokenize(input_path)
    raw_count = sum(len(t) for t in documents.values())

    documents = normalize(documents)
    normalized_count = sum(len(t) for t in documents.values())

    documents = remove_stopwords(documents, stopwords)
    kept_count = sum(len(t) for t in documents.values())

    documents = stem_tokens(documents)
    write_processed(documents, output_path)

    print("documents read          : %d" % len(documents))
    print("raw tokens              : %d" % raw_count)
    print("after normalization     : %d" % normalized_count)
    print("after stopword removal  : %d" % kept_count)
    print("distinct stems          : %d" % len({t for ts in documents.values() for t in ts}))
    print("wrote %s" % output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="cran.all.1400", help="raw Cranfield collection")
    parser.add_argument("--stopwords", default="stopwords.txt", help="stopword list")
    parser.add_argument("--group", default="group", help="group name used as file prefix")
    parser.add_argument("--output", help="output file (default <group>_processed.all)")
    args = parser.parse_args()

    output = args.output or "%s_processed.all" % args.group
    preprocess(args.input, args.stopwords, output)


if __name__ == "__main__":
    main()
