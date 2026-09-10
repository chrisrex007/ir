"""Run the experiment plan: preprocessing, then weighting model, then model parameters.

    python3 search_ninjas_experiments.py            # all three stages
    python3 search_ninjas_experiments.py --stage 3  # one stage, repeatable

Stage 3 writes search_ninjas_experiment_results/best_configuration.json, which
search_ninjas_search.py reads to build the submitted run.
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
RESULTS_DIR = os.path.join(BASE_DIR, "search_ninjas_experiment_results")

METRICS = [AP, nDCG @ 10, nDCG @ 20, P @ 5, P @ 10, R @ 100, RR]
PRIMARY = "AP"

# Sparse vector space weighting models, as the assignment requires.
MODELS = [
    "Tf",            # raw term frequency, no idf
    "TF_IDF",        # Robertson tf x Sparck Jones idf
    "LemurTF_IDF",   # the Lemur variant of the same
    "BM25",          # tf saturation and length normalisation
]

BM25_K1 = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0]
BM25_B = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]
DEFAULTS = {"BM25": {"bm25.k_1": 1.2, "bm25.b": 0.75}}


def load_topics_and_qrels():
    """Return the pyterrier topics and qrels frames for the full query set."""
    topics = pd.DataFrame(
        [{"qid": q["qid"], "query": parse.clean_query(q["query"])} for q in parse.read_queries()]
    )
    qrels = pd.DataFrame([{"qid": q["qid"], "docno": q["docno"], "label": q["label"]} for q in parse.read_qrels()])
    return topics, qrels


def split_topics(topics, qrels):
    """Split odd-id training queries from even-id held-out queries."""
    is_train = topics["qid"].astype(int) % 2 == 1
    train, test = topics[is_train], topics[~is_train]
    return (train, qrels[qrels.qid.isin(train.qid)]), (test, qrels[qrels.qid.isin(test.qid)])


def experiment(systems, names, topics, qrels, extra=None):
    """Run pt.Experiment and optionally splice in extra columns such as timings."""
    frame = pt.Experiment(systems, topics, qrels, eval_metrics=METRICS, names=names, round=4)
    if extra is not None:
        for column, values in extra.items():
            frame.insert(1, column, values)
    return frame


def timed_run(system, topics):
    """Return (total seconds, milliseconds per query) for one retrieval run."""
    start = time.perf_counter()
    system.transform(topics)
    elapsed = time.perf_counter() - start
    return elapsed, 1000.0 * elapsed / len(topics)


def emit(frame, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
    frame.to_csv(csv_path, index=False)
    print(f"\n=== {name} ===")
    print(frame.to_string(index=False))
    print(f"-> {csv_path}")
    return frame


def retriever(index, wmodel, controls):
    # Retriever writes its own controls into the dict it is handed, so pass a copy.
    return pt.terrier.Retriever(index, wmodel=wmodel, controls=dict(controls))


def describe_run(wmodel, controls):
    inner = ", ".join(f"{key.split('.')[-1]}={value}" for key, value in controls.items())
    return f"{wmodel}({inner})"


def stage_preprocessing(topics, qrels):
    """Score every fields x stemmer x stopwords combination with untuned BM25."""
    rows, systems, names, times = [], [], [], []
    for fields in ("text", "title_text", "all"):
        for stemmer in ("none", "weakporter", "porter"):
            for stopwords in ("none", "terrier", "assignment"):
                index, build_time = indexing.build_index(fields, stemmer, stopwords, overwrite=True)
                systems.append(pt.terrier.Retriever(index, wmodel="BM25"))
                names.append(indexing.variant_name(fields, stemmer, stopwords))
                rows.append({"index_time_s": round(build_time, 2), **indexing.describe(index)})
                times.append(round(timed_run(systems[-1], topics)[1], 2))

    frame = experiment(systems, names, topics, qrels)
    detail = pd.DataFrame(rows)
    detail["search_ms_per_query"] = times
    frame = pd.concat([frame.reset_index(drop=True), detail], axis=1)
    return emit(frame.sort_values(PRIMARY, ascending=False), "stage1_preprocessing")


def stage_models(index, topics, qrels):
    """Compare the weighting models at their default parameters."""
    systems = [pt.terrier.Retriever(index, wmodel=model) for model in MODELS]
    times = [round(timed_run(system, topics)[1], 2) for system in systems]
    frame = experiment(systems, MODELS, topics, qrels, extra={"search_ms_per_query": times})
    return emit(frame.sort_values(PRIMARY, ascending=False), "stage2_models")


def stage_parameters(index, train, test, topics, qrels):
    """Sweep BM25's k_1 and b on the training half and compare the winner to the default."""
    # k_1 runs past 2.0 because a narrower grid put the optimum on its boundary.
    grid = [("BM25", {"bm25.k_1": k1, "bm25.b": b}) for k1 in BM25_K1 for b in BM25_B]
    names = [describe_run(model, controls) for model, controls in grid]

    frame = pt.Experiment([retriever(index, m, c) for m, c in grid], train[0], train[1],
                          eval_metrics=METRICS, names=names, round=4)
    frame.insert(1, "model", [model for model, _ in grid])
    emit(frame.sort_values(PRIMARY, ascending=False), "stage3_parameter_sweep_train")

    best_name = frame.sort_values(PRIMARY, ascending=False).iloc[0]["name"]
    best_model, best_controls = dict(zip(names, grid))[best_name]
    print(f"\nbest on train: {best_name}")

    baseline_name = describe_run(best_model, DEFAULTS[best_model]) + " [default]"
    systems = [retriever(index, best_model, DEFAULTS[best_model]), retriever(index, best_model, best_controls)]
    for label, (frame_topics, frame_qrels) in (("heldout", test), ("full", (topics, qrels))):
        out = pt.Experiment(systems, frame_topics, frame_qrels, eval_metrics=METRICS,
                            names=[baseline_name, best_name + " [tuned]"], baseline=0, round=4)
        emit(out, f"stage3_tuned_vs_default_{label}")

    return best_model, best_controls


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, action="append", choices=[1, 2, 3],
                    help="run only these stages (repeatable); default is all three")
    ap.add_argument("--fields", default="all", choices=sorted(indexing.FIELD_SETS))
    ap.add_argument("--stemmer", default="porter", choices=sorted(indexing.STEMMERS))
    ap.add_argument("--stopwords", default="assignment", choices=indexing.STOPWORD_SETS)
    args = ap.parse_args()
    stages = sorted(set(args.stage)) if args.stage else [1, 2, 3]

    topics, qrels = load_topics_and_qrels()
    train, test = split_topics(topics, qrels)
    print(f"{len(topics)} queries ({len(train[0])} train / {len(test[0])} test), {len(qrels)} judged pairs")

    if 1 in stages:
        stage_preprocessing(topics, qrels)

    index, build_time = indexing.build_index(args.fields, args.stemmer, args.stopwords, overwrite=True)
    print(f"\nworking index: {indexing.variant_name(args.fields, args.stemmer, args.stopwords)} "
          f"built in {build_time:.2f} s -> {indexing.describe(index)}")

    if 2 in stages:
        stage_models(index, topics, qrels)

    if 3 in stages:
        best_model, best_controls = stage_parameters(index, train, test, topics, qrels)
        config = {
            "fields": args.fields, "stemmer": args.stemmer, "stopwords": args.stopwords,
            "wmodel": best_model, "controls": best_controls,
            "index_time_s": round(build_time, 2),
            "retrieval_ms_per_query": round(timed_run(retriever(index, best_model, best_controls), topics)[1], 2),
        }
        path = os.path.join(RESULTS_DIR, "best_configuration.json")
        with open(path, "w") as fh:
            json.dump(config, fh, indent=2)
        print(f"\nbest configuration -> {path}\n{json.dumps(config, indent=2)}")


if __name__ == "__main__":
    main()
