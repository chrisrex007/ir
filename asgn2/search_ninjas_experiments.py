"""Run the PA2 experiment plan and write the comparison tables.

Four stages, each one narrowing the next:

  1. preprocessing  -- which fields, stemmer and stopword list to index
  2. models         -- which weighting model on the winning index
  3. parameters     -- sweep the winning model's parameters
  4. expansion      -- pseudo-relevance feedback on top of that

Parameters are tuned on a training half of the 225 queries and reported on the
held-out half as well as on the full set, so a tuned number can be compared
against an untuned one without the tuning having seen the test queries.

    python3 search_ninjas_experiments.py            # every stage
    python3 search_ninjas_experiments.py --stage 3  # just the parameter sweep
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
RESULTS_DIR = os.path.join(BASE_DIR, "search_ninjas_experiment_results")

# ir_measures names; pt.Experiment turns these into columns.
from ir_measures import AP, nDCG, P, R, RR  # noqa: E402

METRICS = [AP, nDCG @ 10, nDCG @ 20, P @ 5, P @ 10, R @ 100, RR]
PRIMARY = "AP"

# Terrier weighting models. The assignment restricts us to sparse lexical
# scoring; these are all term-weighting formulas over the same inverted index,
# with no learned or dense component.
MODELS = [
    "Tf",            # raw term frequency, the degenerate baseline
    "TF_IDF",        # Robertson tf x Sparck Jones idf -- the classic VSM
    "LemurTF_IDF",   # the Lemur/Indri variant of the same
    "BM25",          # Okapi tf saturation + length normalisation
    "DFR_BM25",      # BM25 rebuilt inside the DFR framework
    "PL2",           # DFR: Poisson, Laplace aftereffect, normalisation 2
    "In_expB2",      # DFR: inverse expected document frequency
    "IFB2",          # DFR: inverse term frequency, Bernoulli
    "DPH",           # DFR, parameter-free
    "DFRee",         # DFR, parameter-free
    "DLH13",         # DFR, parameter-free
    "Hiemstra_LM",   # linear-interpolation LM, for reference
]


def load_topics_and_qrels():
    """Return the pyterrier ``topics`` and ``qrels`` frames.

    Query text is stripped to bare alphanumerics: Terrier's query parser reads
    characters like ``/`` and ``.`` as operators, and the Cranfield questions
    are full sentences that end in a period.
    """
    topics = pd.DataFrame(
        [{"qid": q["qid"], "query": clean_query(q["query"])} for q in parse.read_queries()]
    )
    qrels = pd.DataFrame([{"qid": q["qid"], "docno": q["docno"], "label": q["label"]} for q in parse.read_qrels()])
    return topics, qrels


def clean_query(text):
    """Case-fold and reduce to alphanumeric tokens, matching the index pipeline."""
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text.lower()).split())


def split_topics(topics, qrels):
    """Odd query ids train, even query ids test.

    Interleaving rather than cutting the list in half matters here: the
    Cranfield queries are grouped by subject, so a contiguous split would put
    whole topics on one side and tune on a different subject mix than it
    reports on.
    """
    is_train = topics["qid"].astype(int) % 2 == 1
    train, test = topics[is_train], topics[~is_train]
    return (train, qrels[qrels.qid.isin(train.qid)]), (test, qrels[qrels.qid.isin(test.qid)])


def experiment(systems, names, topics, qrels, extra=None):
    frame = pt.Experiment(systems, topics, qrels, eval_metrics=METRICS, names=names, round=4)
    if extra is not None:
        for column, values in extra.items():
            frame.insert(1, column, values)
    return frame


def timed_run(system, topics):
    """Total and per-query wall-clock retrieval time, in seconds and ms."""
    start = time.perf_counter()
    results = system.transform(topics)
    elapsed = time.perf_counter() - start
    return elapsed, 1000.0 * elapsed / len(topics), len(results)


def emit(frame, name):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
    frame.to_csv(csv_path, index=False)
    print(f"\n=== {name} ===")
    print(frame.to_string(index=False))
    print(f"-> {csv_path}")
    return frame


# ---------------------------------------------------------------- stage 1


def stage_preprocessing(topics, qrels):
    """Fields x stemmer x stopwords, scored with untuned BM25."""
    rows, systems, names, times = [], [], [], []
    for fields in ("text", "title_text", "all"):
        for stemmer in ("none", "weakporter", "porter"):
            for stopwords in ("none", "terrier", "assignment"):
                index, build_time = indexing.build_index(fields, stemmer, stopwords, overwrite=True)
                stats = indexing.describe(index)
                systems.append(pt.terrier.Retriever(index, wmodel="BM25"))
                names.append(indexing.variant_name(fields, stemmer, stopwords))
                rows.append({"index_time_s": round(build_time, 2), **stats})
                times.append(round(1000.0 * timed_run(systems[-1], topics)[0] / len(topics), 2))

    frame = experiment(systems, names, topics, qrels)
    detail = pd.DataFrame(rows)
    detail["search_ms_per_query"] = times
    frame = pd.concat([frame.reset_index(drop=True), detail], axis=1)
    return emit(frame.sort_values(PRIMARY, ascending=False), "stage1_preprocessing")


# ---------------------------------------------------------------- stage 2


def stage_models(index, topics, qrels):
    systems = [pt.terrier.Retriever(index, wmodel=model) for model in MODELS]
    times = [round(timed_run(system, topics)[1], 2) for system in systems]
    frame = experiment(systems, MODELS, topics, qrels, extra={"search_ms_per_query": times})
    return emit(frame.sort_values(PRIMARY, ascending=False), "stage2_models")


# ---------------------------------------------------------------- stage 3


# BM25's k1 is swept past 2.0 because an earlier, narrower grid put the optimum
# on the boundary -- a maximum at the edge of a grid is a sign the grid is too
# small, not an answer.
BM25_K1 = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 2.0, 2.5, 3.0, 4.0]
BM25_B = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0]

# The DFR models share one free parameter: c, the term-frequency normalisation.
DFR_C = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]
DFR_MODELS = ["PL2", "In_expB2", "IFB2", "DFR_BM25"]

DEFAULTS = {"BM25": {"bm25.k_1": 1.2, "bm25.b": 0.75}, "PL2": {"c": 1.0},
            "In_expB2": {"c": 1.0}, "IFB2": {"c": 1.0}, "DFR_BM25": {"c": 1.0}}


def retriever(index, wmodel, controls, **kwargs):
    # The dict is copied because Retriever writes its own controls (decorate_batch
    # and friends) into whatever it is handed -- passing the caller's dict lets
    # Terrier's internals leak into the saved configuration and the run labels.
    return pt.terrier.Retriever(index, wmodel=wmodel, controls=dict(controls), **kwargs)


def describe_run(wmodel, controls):
    inner = ", ".join(f"{key.split('.')[-1]}={value}" for key, value in controls.items())
    return f"{wmodel}({inner})"


def stage_parameters(index, train, test, topics, qrels):
    """Sweep every parametrised model on the training half; return the best.

    Both the parameters *and* the choice of model are settled here rather than
    inheriting stage 2's winner, because stage 2 compared models at their
    default parameters and a model can win or lose that comparison purely on
    how well its default happens to suit a collection of 1400 short abstracts.
    """
    grid = [("BM25", {"bm25.k_1": k1, "bm25.b": b}) for k1 in BM25_K1 for b in BM25_B]
    grid += [(model, {"c": c}) for model in DFR_MODELS for c in DFR_C]

    names = [describe_run(model, controls) for model, controls in grid]
    frame = pt.Experiment([retriever(index, m, c) for m, c in grid], train[0], train[1],
                          eval_metrics=METRICS, names=names, round=4)
    frame.insert(1, "model", [model for model, _ in grid])
    emit(frame.sort_values(PRIMARY, ascending=False), "stage3_parameter_sweep_train")

    # Per-model best, so the report can show what tuning bought each family.
    per_model = frame.sort_values(PRIMARY, ascending=False).groupby("model", as_index=False).first()
    emit(per_model.sort_values(PRIMARY, ascending=False), "stage3_best_per_model_train")

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


# ---------------------------------------------------------------- stage 4


def stage_expansion(index, wmodel, controls, train, test, topics, qrels):
    """Pseudo-relevance feedback over the tuned run from stage 3."""
    base = retriever(index, wmodel, controls)
    base_name = describe_run(wmodel, controls)

    configs = []
    for expander in ("Bo1", "KL"):
        for docs in (3, 5, 10):
            for terms in (5, 10, 20):
                qe = pt.rewrite.Bo1QueryExpansion(index, fb_docs=docs, fb_terms=terms) if expander == "Bo1" \
                    else pt.rewrite.KLQueryExpansion(index, fb_docs=docs, fb_terms=terms)
                configs.append((f"{expander}(docs={docs}, terms={terms})", base >> qe >> base))

    frame = pt.Experiment([c[1] for c in configs], train[0], train[1], eval_metrics=METRICS,
                          names=[c[0] for c in configs], round=4)
    emit(frame.sort_values(PRIMARY, ascending=False), "stage4_expansion_train")

    best_name = frame.sort_values(PRIMARY, ascending=False).iloc[0]["name"]
    best_system = dict(configs)[best_name]

    for label, (frame_topics, frame_qrels) in (("heldout", test), ("full", (topics, qrels))):
        out = pt.Experiment(
            [base, best_system], frame_topics, frame_qrels, eval_metrics=METRICS,
            names=[base_name, f"{base_name} + {best_name}"], baseline=0, round=4,
        )
        emit(out, f"stage4_expansion_{label}")

    times = {
        "retrieval_ms_per_query": round(timed_run(base, topics)[1], 2),
        "retrieval_with_expansion_ms_per_query": round(timed_run(best_system, topics)[1], 2),
    }
    print("\nsearch time:", times)
    return best_name, times


# ----------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, action="append", choices=[1, 2, 3, 4],
                    help="run only these stages (repeatable); default is all four")
    ap.add_argument("--fields", default="all", choices=sorted(indexing.FIELD_SETS))
    ap.add_argument("--stemmer", default="porter", choices=sorted(indexing.STEMMERS))
    ap.add_argument("--stopwords", default="assignment", choices=indexing.STOPWORD_SETS)
    args = ap.parse_args()
    stages = sorted(set(args.stage)) if args.stage else [1, 2, 3, 4]

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

    best_model, best_controls = "BM25", DEFAULTS["BM25"]
    if 3 in stages:
        best_model, best_controls = stage_parameters(index, train, test, topics, qrels)

    if 4 in stages:
        best_qe, times = stage_expansion(index, best_model, best_controls, train, test, topics, qrels)
        config = {
            "fields": args.fields, "stemmer": args.stemmer, "stopwords": args.stopwords,
            "wmodel": best_model, "controls": best_controls,
            "expansion": best_qe, "index_time_s": round(build_time, 2), **times,
        }
        path = os.path.join(RESULTS_DIR, "best_configuration.json")
        with open(path, "w") as fh:
            json.dump(config, fh, indent=2)
        print(f"\nbest configuration -> {path}\n{json.dumps(config, indent=2)}")


if __name__ == "__main__":
    main()
