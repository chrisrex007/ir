"""Run the tuned retrieval pipeline over a query file and write a TREC run.

    python3 search_ninjas_search.py                      # cran.qry -> search_ninjas_results.txt
    python3 search_ninjas_search.py --queries other.qry  # any file in cran.qry format
    python3 search_ninjas_search.py --evaluate           # also score the run against cranqrel

Two run files are written because cran.qry's own .I labels and the positional
numbering cranqrel uses disagree; see the README for which to score against what.
"""

import argparse
import json
import os
import time

import search_ninjas_env  # noqa: F401  -- must precede pyterrier, sets JAVA_HOME
import pandas as pd
import pyterrier as pt
from ir_measures import AP, nDCG, P, R, RR  # noqa: E402

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
    "controls": {"bm25.k_1": 2.5, "bm25.b": 0.75},
}


def load_config(path=CONFIG_FILE):
    """Read the tuned configuration written by stage 3, falling back to the defaults."""
    config = dict(DEFAULT_CONFIG)
    if os.path.exists(path):
        with open(path) as fh:
            config.update(json.load(fh))
    return config


def write_run(results, queries, path, key):
    """Write the six-column TREC run format: qid Q0 docno rank score tag."""
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
    topics = pd.DataFrame([{"qid": q["qid"], "query": parse.clean_query(q["query"])} for q in queries])
    retriever = pt.terrier.Retriever(index, wmodel=config["wmodel"],
                                     controls=dict(config["controls"]), num_results=args.top_k)

    start = time.perf_counter()
    results = retriever.transform(topics)
    elapsed = time.perf_counter() - start
    print(f"retrieved {len(results)} results for {len(topics)} queries in {elapsed:.2f} s "
          f"({1000 * elapsed / len(topics):.1f} ms/query)")

    written = write_run(results, queries, args.output, "original_id")
    print(f"-> {args.output} ({written} lines, ids from the query file's .I labels)")

    seq_path = args.output.replace(".txt", "_seqid.txt")
    written = write_run(results, queries, seq_path, "qid")
    print(f"-> {seq_path} ({written} lines, ids 1..{len(topics)} by position)")

    if args.evaluate:
        qrels = pd.DataFrame([{"qid": q["qid"], "docno": q["docno"], "label": q["label"]} for q in parse.read_qrels()])
        frame = pt.Experiment([results], topics, qrels, names=[RUN_TAG], round=4,
                              eval_metrics=[AP, nDCG @ 10, nDCG @ 20, P @ 5, P @ 10, R @ 100, RR])
        print("\n=== evaluation against cranqrel ===")
        print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
