#!/usr/bin/env python3
"""Boolean retrieval over the Cranfield index: "<word1> AND|OR <word2>".

The query words are normalized and stemmed exactly as the collection was, their
postings lists are fetched from the index file, and the two lists are combined
with a linear merge. Matching docids are written to search_ninjas_results.txt.

Because the index file is sorted lexicographically, a term is located with a
binary search over the file's byte offsets (O(log n) seeks) instead of scanning
it, and the merge of two sorted postings lists costs O(len(p1) + len(p2)).

Usage:
    python3 search_ninjas_search.py "aerodynamic AND experimental"
    python3 search_ninjas_search.py --queries queries.txt
"""

import argparse
import sys

import search_ninjas_preprocess
import search_ninjas_porter

INDEX_FILE = "search_ninjas_cran.index"
RESULTS_FILE = "search_ninjas_results.txt"

OPERATORS = ("AND", "OR")


def prepare_term(word):
    """Normalize and stem a query word the same way the collection was processed."""
    pieces = search_ninjas_preprocess.normalize({0: [word]})[0]
    if not pieces:
        return ""
    # A hyphenated query word normalizes to several pieces; use the first.
    return search_ninjas_porter.stem(pieces[0])


def parse_query(query):
    """Split "word1 AND word2" into (word1, operator, word2)."""
    parts = query.split()
    if len(parts) != 3:
        raise ValueError("query must be of the form '<word1> AND|OR <word2>'")
    left, operator, right = parts
    operator = operator.upper()
    if operator not in OPERATORS:
        raise ValueError("operator must be AND or OR, not %r" % parts[1])
    return left, operator, right


def open_index(path):
    """Open the index file and return (handle, data_start, file_size).

    data_start is the offset of the first term line, i.e. just past the header
    line holding the vocabulary size and the maximum docid.
    """
    handle = open(path, "rb")
    handle.readline()  # header: "<vocabulary size>, <maximum docid>"
    data_start = handle.tell()
    handle.seek(0, 2)
    file_size = handle.tell()
    return handle, data_start, file_size


def _key(line):
    """The indexed term at the start of an index line."""
    return line.split(b" ", 1)[0]


def _postings_of(line):
    """Parse "term d1,d2,d3" into a list of docids."""
    parts = line.decode("utf-8").rstrip("\n").split(" ", 1)
    if len(parts) < 2 or not parts[1]:
        return []
    return [int(d) for d in parts[1].split(",") if d.strip()]


def lookup(handle, data_start, file_size, term):
    """Binary search the sorted index file for `term`, returning its postings list."""
    if not term:
        return []
    target = term.encode("utf-8")
    low, high = data_start, file_size

    # Halve the byte range until it closes. Seeking to an arbitrary offset lands
    # in the middle of a line, so the remainder of that line is skipped before a
    # term is read; the offset of a skipped line therefore has to stay inside the
    # range. `low` only ever moves past a line that sorts before the target, so
    # the invariant held throughout is that every line starting below `low` sorts
    # before the target.
    while low < high:
        middle = (low + high) // 2
        handle.seek(middle)
        if middle > data_start:
            handle.readline()
        position = handle.tell()
        line = handle.readline()

        if not line or position >= high:
            high = middle
        elif _key(line) < target:
            low = position + len(line)
        else:
            high = middle

    # The target, if present, is now at or just after `low`: read forward until
    # the term is found or the terms sort past it.
    handle.seek(low)
    while True:
        line = handle.readline()
        if not line:
            return []
        key = _key(line)
        if key == target:
            return _postings_of(line)
        if key > target:
            return []


def intersect(first, second):
    """AND: linear merge keeping the docids present in both postings lists."""
    result = []
    i = j = 0
    while i < len(first) and j < len(second):
        if first[i] == second[j]:
            result.append(first[i])
            i += 1
            j += 1
        elif first[i] < second[j]:
            i += 1
        else:
            j += 1
    return result


def union(first, second):
    """OR: linear merge keeping every docid from either postings list, once."""
    result = []
    i = j = 0
    while i < len(first) and j < len(second):
        if first[i] == second[j]:
            result.append(first[i])
            i += 1
            j += 1
        elif first[i] < second[j]:
            result.append(first[i])
            i += 1
        else:
            result.append(second[j])
            j += 1
    result.extend(first[i:])
    result.extend(second[j:])
    return result


def search(handle, data_start, file_size, query):
    """Answer one Boolean query, returning (stemmed query text, docid list)."""
    left, operator, right = parse_query(query)
    left_term, right_term = prepare_term(left), prepare_term(right)

    left_postings = lookup(handle, data_start, file_size, left_term)
    right_postings = lookup(handle, data_start, file_size, right_term)

    if operator == "AND":
        docids = intersect(left_postings, right_postings)
    else:
        docids = union(left_postings, right_postings)

    return "%s %s %s" % (left_term, operator, right_term), docids


def format_result(query, stemmed, docids):
    """Render one query's answer as the block written to the results file."""
    return "\n".join([
        "Query: %s" % query,
        "Stemmed query: %s" % stemmed,
        "Matched documents: %d" % len(docids),
        ",".join(str(d) for d in docids),
        "",
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", help='e.g. "aerodynamic AND experimental"')
    parser.add_argument("--queries", help="file holding one Boolean query per line")
    parser.add_argument("--index", default=INDEX_FILE, help="index file")
    parser.add_argument("--output", default=RESULTS_FILE, help="results file")
    args = parser.parse_args()

    if not args.query and not args.queries:
        parser.error("give a query, or --queries with a file of queries")

    index_path = args.index
    output_path = args.output

    queries = []
    if args.query:
        queries.append(args.query)
    if args.queries:
        with open(args.queries, encoding="utf-8") as handle:
            queries.extend(line.strip() for line in handle if line.strip())

    handle, data_start, file_size = open_index(index_path)
    blocks = []
    try:
        for query in queries:
            try:
                stemmed, docids = search(handle, data_start, file_size, query)
            except ValueError as error:
                print("skipping %r: %s" % (query, error), file=sys.stderr)
                continue
            blocks.append(format_result(query, stemmed, docids))
            print("%s -> %d documents" % (query, len(docids)))
    finally:
        handle.close()

    if not blocks:
        print("no query could be answered; %s left unchanged" % output_path, file=sys.stderr)
        return 1

    with open(output_path, "w", encoding="utf-8") as out:
        out.write("\n".join(blocks))
    print("wrote %s" % output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
