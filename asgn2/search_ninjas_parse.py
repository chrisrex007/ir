"""Readers for the three Cranfield files: documents, queries and relevance judgements."""

import os
import re

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_FILE = os.path.join(DATA_DIR, "cran.all.1400")
QUERY_FILE = os.path.join(DATA_DIR, "cran.qry")
QREL_FILE = os.path.join(DATA_DIR, "cranqrel")

# Cranfield grades 1 (best) to 4 (worst), so the codes must be flipped into gains.
GAIN_OF_CODE = {1: 4, 2: 3, 3: 2, 4: 1}

TAG_RE = re.compile(r"^\.([IWTAB])\s*(.*)$")


def read_documents(path=DOCS_FILE):
    """Parse cran.all.1400 into a list of {docno, title, author, bib, text}."""
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
                line = rest
            if current is not None and field is not None and line.strip():
                current[field].append(line.strip())

    if current is not None:
        docs.append(_finish(current))
    return docs


def _field_after(tag, field):
    """Return the field a tag opens, or None if it is stray text inside an abstract."""
    # Records run .T .A .B .W, so an .A or .B after .W is abstract text (document 240).
    names = {"T": "title", "A": "author", "B": "bib", "W": "text"}
    if tag == "W":
        return "text"
    if field == "text":
        return None
    return names[tag]


def _finish(doc):
    return {key: " ".join(value) if isinstance(value, list) else value for key, value in doc.items()}


def read_queries(path=QUERY_FILE):
    """Parse a cran.qry-format file into a list of {qid, original_id, query}."""
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
    # cranqrel numbers queries by position, not by the non-contiguous .I labels.
    return {"qid": str(position), "original_id": original_id, "query": " ".join(lines)}


def read_qrels(path=QREL_FILE):
    """Parse cranqrel into a list of {qid, docno, label} with labels as gains."""
    qrels = []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 3:
                continue
            qid, docno, code = parts[0], str(int(parts[1])), int(parts[2])
            if code not in GAIN_OF_CODE:
                continue  # code -1 marks an unjudged pair
            qrels.append({"qid": qid, "docno": docno, "label": GAIN_OF_CODE[code]})
    return qrels


def clean_query(text):
    """Case-fold and reduce a query to alphanumeric tokens, matching the index pipeline."""
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


if __name__ == "__main__":
    docs = read_documents()
    queries = read_queries()
    qrels = read_qrels()
    print(f"documents : {len(docs)}")
    print(f"queries   : {len(queries)}  (labels {queries[0]['original_id']}..{queries[-1]['original_id']})")
    print(f"qrels     : {len(qrels)} judged pairs over {len({q['qid'] for q in qrels})} queries")
