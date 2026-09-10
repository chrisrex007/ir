"""Readers for the three Cranfield files: documents, queries and qrels.

Everything downstream (indexing, experiments, search) gets its data from here so
that the collection's quirks are handled in exactly one place.
"""

import os
import re

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FILE = os.path.join(DATA_DIR, "cran.all.1400")
QUERY_FILE = os.path.join(DATA_DIR, "cran.qry")
QREL_FILE = os.path.join(DATA_DIR, "cranqrel")

# Cleverdon graded the judgements 1 (complete answer) .. 4 (minimum interest),
# i.e. *smaller is better*, which is the reverse of what every evaluation tool
# expects. Flip it so 1 -> 4 .. 4 -> 1 and a missing pair stays 0.
GAIN_OF_CODE = {1: 4, 2: 3, 3: 2, 4: 1}

TAG_RE = re.compile(r"^\.([IWTAB])\s*(.*)$")


def read_documents(path=DOCS_FILE):
    """Parse cran.all.1400 into ``[{docno, title, author, bib, text}]``.

    The file is not uniformly tagged, so the parser is a state machine keyed on
    the last field tag with one extra rule (see ``_field_after``).
    """
    docs = []
    current = None
    field = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            match = TAG_RE.match(line)
            if match:
                tag, rest = match.group(1), match.group(2).strip()
                if tag == "I":
                    if current is not None:
                        docs.append(_finish(current))
                    current = {"docno": str(int(rest)), "title": [], "author": [], "bib": [], "text": []}
                    field = None
                    continue
                new_field = _field_after(tag, field)
                if new_field is not None:
                    field = new_field
                    continue
                # A stray .A/.B inside an abstract: keep it as abstract text.
                line = rest
            if current is not None and field is not None and line.strip():
                current[field].append(line.strip())

    if current is not None:
        docs.append(_finish(current))
    return docs


def _field_after(tag, field):
    """Which record field ``tag`` opens, or None if it is stray text.

    A Cranfield record is written ``.T .A .B .W``. Once ``.W`` has opened, an
    ``.A``/``.B`` line is stray text inside the abstract, not a new field --
    document 240 loses ~15 lines of its abstract without this rule. A second
    ``.W`` (documents 576 and 578) simply continues the abstract.
    """
    names = {"T": "title", "A": "author", "B": "bib", "W": "text"}
    if tag == "W":
        return "text"
    if field == "text":
        return None
    return names[tag]


def _finish(doc):
    return {key: " ".join(value) if isinstance(value, list) else value for key, value in doc.items()}


def read_queries(path=QUERY_FILE):
    """Parse a cran.qry-format file into ``[{qid, query, original_id}]``.

    ``cranqrel`` numbers the queries 1..225 by their *position* in cran.qry,
    while the ``.I`` labels in cran.qry itself are non-contiguous (001, 002,
    004, 008, ...). ``qid`` is therefore the position, which is what the qrels
    and every evaluation script key on; the file's own label is kept as
    ``original_id`` so results can be reported either way.
    """
    queries = []
    text = None
    original_id = None

    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith(".I"):
                if text is not None:
                    queries.append(_finish_query(len(queries) + 1, original_id, text))
                original_id = line[2:].strip()
                text = None
            elif line.startswith(".W"):
                text = []
            elif text is not None and line.strip():
                text.append(line.strip())

    if text is not None:
        queries.append(_finish_query(len(queries) + 1, original_id, text))
    return queries


def _finish_query(position, original_id, lines):
    return {"qid": str(position), "original_id": original_id, "query": " ".join(lines)}


def read_qrels(path=QREL_FILE):
    """Parse cranqrel into ``[{qid, docno, label, code}]``.

    ``label`` is the flipped gain used for evaluation, ``code`` the raw
    Cleverdon grade. Rows coded ``-1`` (one per query, an unjudged marker) are
    dropped -- keeping them would score a document as relevant for having no
    judgement at all.
    """
    qrels = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                continue
            qid, docno, code = parts[0], str(int(parts[1])), int(parts[2])
            if code not in GAIN_OF_CODE:
                continue
            qrels.append({"qid": qid, "docno": docno, "label": GAIN_OF_CODE[code], "code": code})
    return qrels


if __name__ == "__main__":
    docs = read_documents()
    queries = read_queries()
    qrels = read_qrels()
    print(f"documents : {len(docs)}")
    print(f"queries   : {len(queries)}  (labels {queries[0]['original_id']}..{queries[-1]['original_id']})")
    print(f"qrels     : {len(qrels)} judged pairs over {len({q['qid'] for q in qrels})} queries")
    max_qrel_doc = max(int(q["docno"]) for q in qrels)
    print(f"max docno in qrels: {max_qrel_doc} (collection has {len(docs)})")
