#!/usr/bin/env python3
"""Build the inverted index for the preprocessed Cranfield collection.

Reads <group>_processed.all and writes <group>_cran.index. The first line of the
index holds the vocabulary size and the largest indexed docid; every following
line holds one term and its postings list of ascending docids, and the terms
themselves are in lexicographical order.

Usage:
    python3 group_index.py [--input group_processed.all] [--group group]
"""

import argparse
import re

_DOC_TAG = re.compile(r"^\.I\s+(\d+)\s*$")


def read_processed(input_path):
    """Read the processed collection into a dict of docid -> token list."""
    documents = {}
    docid = None
    in_tokens = False

    with open(input_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")

            doc_match = _DOC_TAG.match(line)
            if doc_match:
                docid = int(doc_match.group(1))
                documents[docid] = []
                in_tokens = False
                continue

            if line.strip() == ".S":
                in_tokens = True
                continue

            if docid is not None and in_tokens:
                documents[docid].extend(line.split())

    return documents


def build_index(documents):
    """Invert the collection into a dict of term -> sorted list of docids."""
    postings = {}
    for docid in sorted(documents):
        for token in documents[docid]:
            docids = postings.setdefault(token, [])
            # Documents are visited in ascending order, so a term's postings list
            # stays sorted and a duplicate can only be the docid just appended.
            if not docids or docids[-1] != docid:
                docids.append(docid)
    return postings


def write_index(postings, max_docid, output_path):
    """Write the index file: header line, then one sorted term per line."""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("%d, %d\n" % (len(postings), max_docid))
        for term in sorted(postings):
            handle.write("%s %s\n" % (term, ",".join(str(d) for d in postings[term])))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="processed collection (default <group>_processed.all)")
    parser.add_argument("--group", default="group", help="group name used as file prefix")
    parser.add_argument("--output", help="index file (default <group>_cran.index)")
    args = parser.parse_args()

    input_path = args.input or "%s_processed.all" % args.group
    output_path = args.output or "%s_cran.index" % args.group

    documents = read_processed(input_path)
    postings = build_index(documents)
    max_docid = max(documents) if documents else 0
    write_index(postings, max_docid, output_path)

    total_postings = sum(len(d) for d in postings.values())
    print("vocabulary size  : %d" % len(postings))
    print("maximum docid    : %d" % max_docid)
    print("total postings   : %d" % total_postings)
    print("wrote %s" % output_path)


if __name__ == "__main__":
    main()
