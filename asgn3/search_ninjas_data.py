"""Cranfield parsing and deterministic text preprocessing for assignment 3."""
import re
from pathlib import Path
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

FIELDS = ('title', 'author', 'bibliography', 'body')


def tokens(text):
    """Lowercase alphanumeric tokens, English stopwords removed, no stemming."""
    return [t for t in re.findall(r'[a-z0-9]+', text.lower())
            if t not in ENGLISH_STOP_WORDS]


def read_documents(path):
    documents, doc, field = [], None, None
    tags = {'T': 'title', 'A': 'author', 'B': 'bibliography', 'W': 'body'}
    for line in Path(path).read_text().splitlines():
        match = re.fullmatch(r'\.([ITABW])(?:\s+(.*))?', line.strip())
        if match:
            tag, value = match.groups()
            if tag == 'I':
                if doc is not None:
                    documents.append(doc)
                doc = {'docno': str(int(value)), **{f: '' for f in FIELDS}}
                field = None
            elif field == 'body' and tag in ('A', 'B'):
                # Stray tags inside document 240 must not discard its abstract.
                if value:
                    doc[field] += ' ' + value
            else:
                field = tags[tag]
                if value:
                    doc[field] += ' ' + value
        elif doc is not None and field:
            doc[field] += ' ' + line.strip()
    if doc is not None:
        documents.append(doc)
    ids = [d['docno'] for d in documents]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Document collection is empty or contains duplicate ids')
    return sorted(documents, key=lambda d: int(d['docno']))


def read_queries(path):
    queries, current, in_text = [], None, False
    for line in Path(path).read_text().splitlines():
        if line.startswith('.I '):
            current = {'qid': str(len(queries) + 1),
                       'original_id': line[3:].strip(), 'text': ''}
            queries.append(current)
            in_text = False
        elif line.strip() == '.W':
            in_text = True
        elif current is not None and in_text:
            current['text'] += ' ' + line.strip()
    labels = [q['original_id'] for q in queries]
    if not queries or len(labels) != len(set(labels)):
        raise ValueError('Query file is empty or contains duplicate labels')
    return queries


def read_qrels(path):
    """Codes 1..4 are relevant; -1 markers and code 5 are not relevant."""
    relevant = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        qid, docno, code = line.split()
        code = int(code)
        if code not in (-1, 1, 2, 3, 4, 5):
            raise ValueError(f'Unknown relevance code: {code}')
        relevant.setdefault(str(int(qid)), set())
        if 1 <= code <= 4:
            relevant[str(int(qid))].add(str(int(docno)))
    return relevant
