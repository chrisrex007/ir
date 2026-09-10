"""Run the tuned retrieval pipeline over a query file and write a TREC run.

    python3 search_ninjas_search.py                          # cran.qry -> search_ninjas_results.txt
    python3 search_ninjas_search.py --queries other.qry      # any file in cran.qry format
    python3 search_ninjas_search.py --evaluate               # also score against cranqrel

The configuration comes from ``search_ninjas_experiment_results/best_configuration.json``
if the experiment driver has been run, otherwise from the defaults below, so
this script always reflects the tuning rather than restating it.

Query ids in the output
-----------------------
``cranqrel`` numbers the 225 queries 1..225 by their position in cran.qry, but
cran.qry's own ``.I`` labels are non-contiguous (001, 002, 004, 008, ...). The
two numberings therefore disagree, and which one a run file should carry
depends on the judgements it will be scored against. Rather than guess, every
run is written twice: ``search_ninjas_results.txt`` keyed on the query file's
own ``.I`` labels, and ``search_ninjas_results_seqid.txt`` keyed on position
(1..N), which is what cranqrel expects.
"""

import argparse
import json
import os
import time

import search_ninjas_env  # noqa: F401  -- must precede pyterrier, sets JAVA_HOME
import pandas as pd
import pyterrier as pt

import search_ninjas_index as indexing
import search_ninjas_parse as parse

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "search_ninjas_experiment_results", "best_configuration.json")
RESULTS_FILE = os.path.join(BASE_DIR, "search_ninjas_results.txt")
RUN_TAG = "search_ninjas"

DEFAULT_CONFIG = {
    "fields": "all",
    "stemmer": "porter",
    "stopwords": "assignment",
    "wmodel": "BM25",
    "controls": {"bm25.k_1": 1.2, "bm25.b": 0.75},
    "expansion": None,
}


def load_config(path=CONFIG_FILE):
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path) as fh:
            config.update(json.load(fh))
    return config


def build_pipeline(config, index):
    """Assemble retrieval (+ optional pseudo-relevance feedback) from the config."""
    # dict(...) because Retriever writes its own controls into the dict it is given.
    retriever = pt.terrier.Retriever(index, wmodel=config["wmodel"],
                                     controls=dict(config.get("controls", {})), num_results=1000)

    expansion = config.get("expansion")
    if not expansion:
        return retriever

    # e.g. "Bo1(docs=10, terms=20)"
    kind, _, rest = expansion.partition("(")
    params = dict(part.split("=") for part in rest.rstrip(")").replace(" ", "").split(",")) if rest else {}
    docs, terms = int(params.get("docs", 3)), int(params.get("terms", 10))
    expander = (pt.rewrite.Bo1QueryExpansion if kind == "Bo1" else pt.rewrite.KLQueryExpansion)(
        index, fb_docs=docs, fb_terms=terms
    )
    return retriever >> expander >> retriever


def clean_query(text):
    """Case-fold and reduce to alphanumeric tokens, matching the index pipeline."""
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def write_run(results, queries, path, key):
    """Write TREC six-column run format: ``qid Q0 docno rank score tag``."""
    label_of = {q["qid"]: q[key] for q in queries}
    lines = 0
    with open(path, "w") as fh:
        for row in results.sort_values(["qid", "rank"]).itertuples():
            fh.write(f"{label_of[row.qid]} Q0 {row.docno} {row.rank + 1} {row.score:.6f} {RUN_TAG}\n")
            lines += 1
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queries", default=parse.QUERY_FILE, help="query file in cran.qry format")
    ap.add_argument("--output", default=RESULTS_FILE)
    ap.add_argument("--top-k", type=int, default=100, help="results per query to write (default 100)")
    ap.add_argument("--evaluate", action="store_true", help="score the run against cranqrel")
    args = ap.parse_args()

    config = load_config()
    print("configuration:", json.dumps(config, indent=2))

    index, index_time = indexing.build_index(config["fields"], config["stemmer"], config["stopwords"])
    print(f"index: {indexing.describe(index)}"
          + (f"  (built in {index_time:.2f} s)" if index_time else "  (reused)"))

    queries = parse.read_queries(args.queries)
    topics = pd.DataFrame([{"qid": q["qid"], "query": clean_query(q["query"])} for q in queries])
    pipeline = build_pipeline(config, index)

    start = time.perf_counter()
    results = pipeline.transform(topics)
    elapsed = time.perf_counter() - start
    print(f"retrieved {len(results)} results for {len(topics)} queries in {elapsed:.2f} s "
          f"({1000 * elapsed / len(topics):.1f} ms/query)")

    results = results[results["rank"] < args.top_k]
    written = write_run(results, queries, args.output, "original_id")
    print(f"-> {args.output} ({written} lines, ids from the query file's .I labels)")

    seq_path = args.output.replace(".txt", "_seqid.txt")
    written = write_run(results, queries, seq_path, "qid")
    print(f"-> {seq_path} ({written} lines, ids 1..{len(topics)} by position, as cranqrel numbers them)")

    if args.evaluate:
        qrels = pd.DataFrame([{"qid": q["qid"], "docno": q["docno"], "label": q["label"]} for q in parse.read_qrels()])
        from ir_measures import AP, nDCG, P, R, RR

        frame = pt.Experiment([results], topics, qrels, names=[RUN_TAG], round=4,
                              eval_metrics=[AP, nDCG @ 10, nDCG @ 20, P @ 5, P @ 10, R @ 100, RR])
        print("\n=== evaluation against cranqrel ===")
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
