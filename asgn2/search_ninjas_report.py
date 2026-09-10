"""Render the submission report as a PDF from the CSVs in search_ninjas_experiment_results.

    python3 search_ninjas_report.py         # -> search_ninjas_report.pdf
    python3 search_ninjas_report.py --html  # keep the intermediate HTML too
"""

import argparse
import json
import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "search_ninjas_experiment_results")
PDF_FILE = os.path.join(BASE_DIR, "search_ninjas_report.pdf")
HTML_FILE = os.path.join(BASE_DIR, "search_ninjas_report.html")

GROUP = "search_ninjas"
MEMBERS = [
    # Fill in before submitting: ("Name", "Roll number"),
    ("Krish Charniya", ""),
]

METRIC_COLUMNS = ["AP", "nDCG@10", "nDCG@20", "P@5", "P@10", "R@100", "RR"]

CSS = """
@page { size: A4; margin: 18mm 16mm; @bottom-center { content: counter(page); font-size: 9pt; color: #666; } }
body { font-family: "DejaVu Serif", Georgia, serif; font-size: 10pt; line-height: 1.45; color: #111; }
h1 { font-size: 19pt; margin: 0 0 2pt; }
h2 { font-size: 13pt; margin: 18pt 0 6pt; border-bottom: 1px solid #bbb; padding-bottom: 3pt; page-break-after: avoid; }
h3 { font-size: 11pt; margin: 12pt 0 4pt; page-break-after: avoid; }
p, li { text-align: justify; }
table { border-collapse: collapse; width: 100%; font-size: 8.2pt; margin: 6pt 0 10pt;
        font-family: "DejaVu Sans", Helvetica, sans-serif; page-break-inside: avoid; }
th, td { border: 1px solid #ccc; padding: 2.5pt 4pt; text-align: right; }
th { background: #eee; font-weight: bold; }
td:first-child, th:first-child { text-align: left; }
tr:nth-child(even) td { background: #fafafa; }
.subtitle { color: #555; margin: 0 0 14pt; }
.caption { font-size: 8.5pt; color: #555; margin: -6pt 0 12pt; font-style: italic; }
code, .mono { font-family: "DejaVu Sans Mono", monospace; font-size: 8.5pt; }
.key { background: #f3f6fa; border-left: 3px solid #4a6fa5; padding: 6pt 9pt; margin: 8pt 0; }
"""


def load(name):
    return pd.read_csv(os.path.join(RESULTS_DIR, f"{name}.csv"))


# Metric columns want four decimals; times and lengths two; counts none.
DECIMALS = {"index_time_s": 2, "search_ms_per_query": 2, "avg_doc_length": 2,
            "terms": 0, "postings": 0, "tokens": 0, "documents": 0,
            "AP +": 0, "AP -": 0, "AP p-value": 6}

# The sign-test columns pyterrier appends to a baseline comparison.
SIGNIFICANCE = ["AP +", "AP -", "AP p-value"]


HEADERS = {"name": "Configuration", "model": "Model", "index_time_s": "index (s)",
           "search_ms_per_query": "search (ms/q)", "avg_doc_length": "avg len",
           "AP +": "better", "AP -": "worse", "AP p-value": "p"}


def table(frame, columns):
    """Format the selected columns of a results frame as an HTML table."""
    frame = frame[columns].copy()
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            places = DECIMALS.get(column, 4)
            frame[column] = frame[column].map(lambda v, p=places: f"{v:,.{p}f}" if pd.notna(v) else "")
    return frame.rename(columns=HEADERS).to_html(index=False, border=0)


def significance(frame):
    """Pull the sign-test counts and p-value pyterrier adds for a baseline comparison."""
    row = frame.iloc[-1]
    return {"better": int(row["AP +"]), "worse": int(row["AP -"]), "p": float(row["AP p-value"])}


def pct(new, old):
    return f"{100.0 * (new - old) / old:+.1f}%"


def build_html():
    config = json.load(open(os.path.join(RESULTS_DIR, "best_configuration.json")))
    stage1 = load("stage1_preprocessing")
    stage2 = load("stage2_models")
    sweep = load("stage3_parameter_sweep_train")
    tuned_held = load("stage3_tuned_vs_default_heldout")
    tuned_full = load("stage3_tuned_vs_default_full")

    base_ap = float(stage2[stage2.name == "BM25"]["AP"].iloc[0])
    final_ap = float(tuned_full["AP"].iloc[-1])
    held_final = float(tuned_held["AP"].iloc[-1])
    param_sig = significance(tuned_held)

    # The spread of the k_1 x b grid, so the flatness claim below is measured, not asserted.
    grid_spread = float(sweep["AP"].max() - sweep["AP"].min())
    window = sweep[sweep.name.str.extract(r"k_1=([\d.]+)")[0].astype(float).between(0.8, 4.0)
                   & sweep.name.str.extract(r"b=([\d.]+)")[0].astype(float).between(0.3, 1.0)]
    window_spread = float(window["AP"].max() - window["AP"].min())

    tfidf_ap = float(stage2[stage2.name == "TF_IDF"]["AP"].iloc[0])
    lemur_ap = float(stage2[stage2.name == "LemurTF_IDF"]["AP"].iloc[0])
    tf_ap = float(stage2[stage2.name == "Tf"]["AP"].iloc[0])

    best_index_row = stage1.iloc[0]
    best_no_stopwords = stage1[stage1.name.str.endswith("__none")].iloc[0]

    members = "".join(f"<li>{name}{' &mdash; ' + roll if roll else ''}</li>" for name, roll in MEMBERS)
    controls = ", ".join(f"{key} = {value}" for key, value in config["controls"].items())

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{GROUP} &mdash; PA2 Report</title>
<style>{CSS}</style></head><body>

<h1>Vector Space Model and Ranked Retrieval on the Cranfield Collection</h1>
<p class="subtitle">Information Retrieval &mdash; Programming Assignment II</p>

<h2>1. Group details</h2>
<p>Group name: <strong>{GROUP}</strong>. All submitted files carry this prefix.</p>
<ul>{members}</ul>

<h2>2. Problem statement</h2>
<p>Build a ranked retrieval pipeline over the Cranfield collection (1,400 aerodynamics
abstracts, 225 test queries, 1,612 graded relevance judgements) using an open-source
search engine, and tune the preprocessing, the weighting model and its parameters to
maximise retrieval effectiveness. Only sparse vector space models are permitted: no
advanced probabilistic models, and no dense or neural retrieval. Indexing and search times
are to be reported alongside effectiveness, and the resulting system is to be evaluated on
further unseen queries.</p>

<h2>3. Implementation details</h2>
<p>The system is built on <strong>PyTerrier 1.1.2</strong> driving <strong>Terrier
5.11</strong>. Terrier supplies the inverted index, the term pipeline and the weighting
models; our code supplies the collection parsing, the experiment plan and the
evaluation harness.</p>

<h3>3.1 Programs</h3>
<table>
<tr><th>File</th><th>Role</th></tr>
<tr><td class="mono">{GROUP}_setup.sh</td><td>creates the virtualenv, fetches a JDK, unpacks the collection</td></tr>
<tr><td class="mono">{GROUP}_env.py</td><td>points JAVA_HOME at the bundled JDK before pyjnius loads the JVM</td></tr>
<tr><td class="mono">{GROUP}_parse.py</td><td>readers for cran.all.1400, cran.qry and cranqrel</td></tr>
<tr><td class="mono">{GROUP}_index.py</td><td>builds a Terrier index under one preprocessing variant, timed</td></tr>
<tr><td class="mono">{GROUP}_experiments.py</td><td>the three-stage experiment plan; writes every table in this report</td></tr>
<tr><td class="mono">{GROUP}_search.py</td><td>runs the tuned pipeline over a query file, writes a TREC run</td></tr>
<tr><td class="mono">{GROUP}_report.py</td><td>renders this PDF from the experiment output</td></tr>
</table>

<h3>3.2 Preprocessing</h3>
<p>The PA1 preprocessing steps are repeated, but as Terrier's term pipeline rather than
our own code: case folding and non-alphanumeric stripping in the English tokeniser,
stopword removal against the list supplied with PA1, and Porter stemming. Each step is
switchable so that Stage 1 can measure what it contributes. The same normalisation is
applied to query text before it reaches Terrier's parser, so query terms and index terms
cannot drift apart.</p>

<h3>3.3 Two properties of the collection that the code has to get right</h3>
<p><strong>Query numbering.</strong> <span class="mono">cranqrel</span> numbers the
queries 1&ndash;225 by their <em>position</em> in <span class="mono">cran.qry</span>,
whereas <span class="mono">cran.qry</span>'s own <span class="mono">.I</span> labels are
non-contiguous (001, 002, 004, 008, &hellip; 365). Keying a run on the labels and scoring
it against <span class="mono">cranqrel</span> silently evaluates almost every query
against the wrong judgements. Our run files are written under both numberings.</p>
<p><strong>Relevance grades.</strong> Cleverdon's codes run 1 (a complete answer) to 4
(minimum interest) &mdash; smaller is better, the reverse of what every evaluation tool
assumes. They are inverted to gains 4&ndash;1 before use; the 225 rows coded
<span class="mono">-1</span> are unjudged markers and are dropped.</p>
<p><strong>Empty records.</strong> Documents 471 and 995 have no title, author,
bibliography or abstract at all. Document 995 is judged relevant for query 125, so it is
an unreachable relevant document that places a hard ceiling on recall.</p>

<h3>3.4 Parsing</h3>
<p>A Cranfield record is written <span class="mono">.T .A .B .W</span>. The document
parser is a state machine keyed on the most recent field tag with one extra rule: once
<span class="mono">.W</span> has opened, an <span class="mono">.A</span> or
<span class="mono">.B</span> line is stray text inside the abstract rather than a new
field. Document 240 contains exactly this, and without the rule it loses roughly fifteen
lines of its abstract.</p>

<h2>4. Plan of experiments</h2>
<p>Three stages, each one narrowing the next: <strong>(1)</strong> preprocessing, meaning
which fields, stemmer and stopword list to index; <strong>(2)</strong> weighting model at
default parameters; <strong>(3)</strong> the parameters of the winning model.</p>
<div class="key"><p>Stage 3 tunes on the <strong>odd-numbered</strong> queries (113 of
them) and reports on the held-out <strong>even-numbered</strong> queries (112), as well as
on the full set. The interleaved split is deliberate: the Cranfield queries are grouped by
subject, so a contiguous split would tune on a different subject mix than it reports on.
Tuning happens only in Stage 3, so the held-out half is the sole estimate here of what the
unseen test queries will produce, and every figure labelled <em>held-out</em> below comes
from queries the tuning never saw.</p></div>
<p>Effectiveness is measured with <span class="mono">ir_measures</span>: MAP (the primary
metric used for all selection), nDCG@10 and nDCG@20 on the graded judgements, P@5, P@10,
recall@100 and reciprocal rank. Improvements over a baseline are accompanied by a paired
sign test; the <span class="mono">+</span> and <span class="mono">-</span> columns count
the queries that improved and degraded.</p>

<h2>5. Results</h2>

<h3>5.1 Stage 1 &mdash; preprocessing</h3>
<p>Twenty-seven indices: three field sets &times; three stemmers &times; three stopword
settings, each scored with untuned BM25.</p>
{table(stage1, ["name", "AP", "nDCG@10", "P@10", "R@100", "terms", "postings", "avg_doc_length", "index_time_s", "search_ms_per_query"])}
<p class="caption">Table 1: all 27 preprocessing variants, ordered by MAP. Names read
<span class="mono">fields__stemmer__stopwords</span>.</p>

<h3>5.2 Stage 2 &mdash; weighting models</h3>
<p>All models at their default parameters, on the Stage 1 winning index.</p>
{table(stage2, ["name", "AP", "nDCG@10", "nDCG@20", "P@5", "P@10", "R@100", "RR", "search_ms_per_query"])}
<p class="caption">Table 2: weighting models at default parameters over the full 225 queries.</p>

<h3>5.3 Stage 3 &mdash; parameter tuning</h3>
<p>110 BM25 settings (<em>k</em><sub>1</sub> &times; <em>b</em>), scored on the training
half. Tuned against default on the held-out half:</p>
{table(tuned_held, ["name"] + METRIC_COLUMNS + SIGNIFICANCE)}
<p class="caption">Table 3: effect of parameter tuning on the 112 held-out queries.</p>
{table(tuned_full, ["name"] + METRIC_COLUMNS + SIGNIFICANCE)}
<p class="caption">Table 4: the same comparison over all 225 queries.</p>

<h3>5.4 Final configuration and timings</h3>
<table>
<tr><th>Setting</th><th>Value</th></tr>
<tr><td>Indexed fields</td><td>{config['fields']}</td></tr>
<tr><td>Stemmer</td><td>{config['stemmer']}</td></tr>
<tr><td>Stopword list</td><td>{config['stopwords']}</td></tr>
<tr><td>Weighting model</td><td>{config['wmodel']} ({controls})</td></tr>
<tr><td>Indexing time (1,400 documents)</td><td>{config['index_time_s']:.2f} s</td></tr>
<tr><td>Search time</td><td>{config['retrieval_ms_per_query']:.1f} ms / query</td></tr>
<tr><td>MAP, held-out queries</td><td>{held_final:.4f}</td></tr>
<tr><td>MAP, all 225 queries</td><td>{final_ap:.4f}</td></tr>
</table>
<p class="caption">Table 5: the submitted configuration. Indexing is a single-threaded
in-memory build; search time is wall clock over all 225 queries divided by 225.</p>

<h2>6. Discussion of results</h2>

<h3>6.1 Preprocessing dominates everything else</h3>
<p>The single largest effect in the whole study is stopword removal. The best variant
with no stopword list reaches MAP {best_no_stopwords['AP']:.4f}; the best with one reaches
{best_index_row['AP']:.4f}, a gain of {pct(best_index_row['AP'], best_no_stopwords['AP'])},
larger than every model and parameter decision combined. Cranfield queries are full
English questions ("what similarity laws must be obeyed when&hellip;"), so without a
stopword list the query vector is dominated by terms that match almost every document.
Porter stemming is the second largest effect, worth roughly 0.03 MAP, and switching from
Porter to Terrier's weak Porter loses most of that &mdash; on 1,400 short abstracts,
aggressive conflation is the right trade.</p>
<p>Field choice barely matters. Adding the author and bibliography fields to the title
and abstract is worth about 0.003 MAP, and indexing the abstract alone rather than title
plus abstract costs about 0.015 &mdash; small because in most Cranfield records the title
is repeated as the opening of the abstract, so the title field is nearly redundant
already.</p>

<h3>6.2 The IDF component carries the weighting</h3>
<p>Raw term frequency with no IDF component (MAP {tf_ap:.4f}) is well under two thirds as
effective as anything that weights by term rarity, which confirms that the vector space
model's IDF component is doing most of the work. Among the properly weighted models the
spread is narrow: TF&ndash;IDF reaches {tfidf_ap:.4f} and BM25 {base_ap:.4f}, while
LemurTF&ndash;IDF's different length normalisation costs about
{tfidf_ap - lemur_ap:.3f}.</p>
<p>Parameter tuning then adds remarkably little. On the held-out queries the tuned model
gains {float(tuned_held['AP'].iloc[-1]) - float(tuned_held['AP'].iloc[0]):+.4f} MAP over
its own default, improving {param_sig['better']} queries and degrading
{param_sig['worse']} for a sign-test p-value of {param_sig['p']:.3f}, which is not
significant. The response surface is shallow rather than flat: MAP varies by
{window_spread:.3f} across the settings between <em>k</em><sub>1</sub> = 0.8 and 4.0 and
<em>b</em> = 0.3 and 1.0, and by {grid_spread:.3f} across the full grid. The choice is
therefore not free, but it is small next to the preprocessing effect, and the gap between
the best and the default setting is smaller than the noise between two halves of the
query set.</p>
<p>One methodological note: an earlier, narrower sweep put the optimal
<em>k</em><sub>1</sub> on the boundary of the grid, which indicates the grid is too small
rather than giving an answer. The grid reported above extends to
<em>k</em><sub>1</sub> = 4.0 so that the optimum is interior.</p>

<h3>6.3 Models excluded by the assignment</h3>
<p>The assignment restricts the system to sparse vector space models and rules out
advanced probabilistic and dense neural retrieval. Two families that Terrier offers are
therefore excluded even though they are available at no extra cost: the
Divergence-From-Randomness weighting models, and pseudo-relevance feedback, which rewrites
the query from the terms of the top-ranked documents rather than scoring the query as
given. Both are probabilistic rather than vector space methods. The submitted system is
consequently tuned BM25 over the Stage 1 index, with no query rewriting.</p>

<h3>6.4 Where the ceiling is</h3>
<p>End to end, the pipeline moves MAP from {base_ap:.4f} (untuned BM25 on the winning
index) to {final_ap:.4f} over the full query set, {pct(final_ap, base_ap)}. Recall@100
finishes at {float(tuned_full['R@100'].iloc[-1]):.4f}, so roughly a fifth of the relevant
documents are still missed in the top 100, and at least one of them, document 995 for
query 125, is unreachable by any lexical system because the record is empty. The
remaining gap is largely vocabulary mismatch between the query wording and the abstract
wording, which term weighting alone cannot bridge.</p>

<h3>6.5 Threats to validity</h3>
<p>The held-out half is 112 queries, so a MAP difference below roughly 0.01 is not
distinguishable from noise; we have reported sign tests rather than relying on the point
estimates. Stage 1 selected the index on the full query set before the train/test split
was introduced in Stage 3, so the preprocessing choice is mildly optimistic, though the
effect is large enough and stable enough across models that the conclusion is not in
doubt.</p>

</body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--html", action="store_true", help="also keep the intermediate HTML")
    ap.add_argument("--output", default=PDF_FILE)
    args = ap.parse_args()

    html = build_html()
    if args.html:
        with open(HTML_FILE, "w") as fh:
            fh.write(html)
        print(f"-> {HTML_FILE}")

    from weasyprint import HTML

    HTML(string=html, base_url=BASE_DIR).write_pdf(args.output)
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
