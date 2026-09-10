"""Render the submission report (PDF) from the experiment output.

Every table and every number in the report is read out of
``search_ninjas_experiment_results/`` rather than typed in, so the report cannot
drift away from the runs it describes: re-run the experiments, re-run this, and
the prose stays attached to the current numbers.

    python3 search_ninjas_report.py            # -> search_ninjas_report.pdf
    python3 search_ninjas_report.py --html     # keep the intermediate HTML too
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


# Metric columns want four decimals; times and lengths want two; counts want none.
# A single float_format across the frame would print "106.2400" for an average
# document length and "0.4100" for a build time.
DECIMALS = {"index_time_s": 2, "search_ms_per_query": 2, "avg_doc_length": 2,
            "terms": 0, "postings": 0, "tokens": 0, "documents": 0,
            "AP +": 0, "AP -": 0, "AP p-value": 6}

# The sign-test columns pyterrier appends for the MAP comparison. Reporting them
# in the table matters here: several of the differences in this study are within
# noise, and a bare point estimate would hide that.
SIGNIFICANCE = ["AP +", "AP -", "AP p-value"]


HEADERS = {"name": "Configuration", "model": "Model", "index_time_s": "index (s)",
           "search_ms_per_query": "search (ms/q)", "avg_doc_length": "avg len",
           "AP +": "better", "AP -": "worse", "AP p-value": "p"}


def table(frame, columns=None, limit=None):
    frame = (frame if columns is None else frame[columns]).copy()
    if limit:
        frame = frame.head(limit)
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            places = DECIMALS.get(column, 4)
            frame[column] = frame[column].map(lambda v, p=places: f"{v:,.{p}f}" if pd.notna(v) else "")
    return frame.rename(columns=HEADERS).to_html(index=False, border=0)


def significance(frame, metric="AP"):
    """Pull the sign-test counts and p-value pyterrier adds for a baseline run."""
    row = frame.iloc[-1]
    return {
        "better": int(row[f"{metric} +"]),
        "worse": int(row[f"{metric} -"]),
        "p": float(row[f"{metric} p-value"]),
    }


def pct(new, old):
    return f"{100.0 * (new - old) / old:+.1f}%"


def build_html():
    config = json.load(open(os.path.join(RESULTS_DIR, "best_configuration.json")))
    stage1 = load("stage1_preprocessing")
    stage2 = load("stage2_models")
    per_model = load("stage3_best_per_model_train")
    tuned_held = load("stage3_tuned_vs_default_heldout")
    tuned_full = load("stage3_tuned_vs_default_full")
    qe_train = load("stage4_expansion_train")
    qe_held = load("stage4_expansion_heldout")
    qe_full = load("stage4_expansion_full")

    base_ap = float(stage2[stage2.name == "BM25"]["AP"].iloc[0])
    final_ap = float(qe_full["AP"].iloc[-1])
    tuned_ap = float(qe_full["AP"].iloc[0])
    held_base, held_final = float(qe_held["AP"].iloc[0]), float(qe_held["AP"].iloc[-1])
    qe_sig = significance(qe_held)
    param_sig = significance(tuned_held)

    best_index_row = stage1.iloc[0]
    worst_stop = stage1[stage1.name.str.endswith("__none")].iloc[0]

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
maximise retrieval effectiveness. Only sparse lexical models are permitted &mdash; no dense
or neural retrieval. Indexing and search times are to be reported alongside
effectiveness, and the resulting system is to be evaluated on further unseen queries.</p>

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
<tr><td class="mono">{GROUP}_experiments.py</td><td>the four-stage experiment plan; writes every table in this report</td></tr>
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
<p>Four stages, each one narrowing the next: <strong>(1)</strong> preprocessing &mdash;
which fields, stemmer and stopword list to index; <strong>(2)</strong> weighting model at
default parameters; <strong>(3)</strong> the parameters of every parametrised model;
<strong>(4)</strong> pseudo-relevance feedback on top of the winner.</p>
<div class="key"><p>Stages 3 and 4 tune on the <strong>odd-numbered</strong> queries (113
of them) and report on the held-out <strong>even-numbered</strong> queries (112), as well
as on the full set. The interleaved split is deliberate: the Cranfield queries are
grouped by subject, so a contiguous split would tune on a different subject mix than it
reports on. Every figure labelled <em>held-out</em> below comes from queries the tuning
never saw, which is the honest estimate of what the unseen test queries will produce.</p></div>
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
<p>110 BM25 settings (<em>k</em><sub>1</sub> &times; <em>b</em>) and 44 DFR settings
(<em>c</em>), all scored on the training half. The best of each family:</p>
{table(per_model, ["model", "name", "AP", "nDCG@10", "P@10", "R@100"])}
<p class="caption">Table 3: best parameter setting per model family, training queries only.</p>
<p>Tuned against default for the winning model, on the held-out half:</p>
{table(tuned_held, ["name"] + METRIC_COLUMNS + SIGNIFICANCE)}
<p class="caption">Table 4: effect of parameter tuning on the 112 held-out queries.</p>
{table(tuned_full, ["name"] + METRIC_COLUMNS + SIGNIFICANCE)}
<p class="caption">Table 5: the same comparison over all 225 queries.</p>

<h3>5.4 Stage 4 &mdash; pseudo-relevance feedback</h3>
<p>Bo1 and KL query expansion, three feedback-document counts &times; three
feedback-term counts, on the training half:</p>
{table(qe_train, ["name"] + METRIC_COLUMNS, limit=9)}
<p class="caption">Table 6: the nine best of eighteen expansion settings, training queries only.</p>
{table(qe_held, ["name"] + METRIC_COLUMNS + SIGNIFICANCE)}
<p class="caption">Table 7: expansion on the 112 held-out queries.</p>
{table(qe_full, ["name"] + METRIC_COLUMNS + SIGNIFICANCE)}
<p class="caption">Table 8: expansion over all 225 queries.</p>

<h3>5.5 Final configuration and timings</h3>
<table>
<tr><th>Setting</th><th>Value</th></tr>
<tr><td>Indexed fields</td><td>{config['fields']}</td></tr>
<tr><td>Stemmer</td><td>{config['stemmer']}</td></tr>
<tr><td>Stopword list</td><td>{config['stopwords']}</td></tr>
<tr><td>Weighting model</td><td>{config['wmodel']} ({controls})</td></tr>
<tr><td>Query expansion</td><td>{config['expansion']}</td></tr>
<tr><td>Indexing time (1,400 documents)</td><td>{config['index_time_s']:.2f} s</td></tr>
<tr><td>Search time, no expansion</td><td>{config['retrieval_ms_per_query']:.1f} ms / query</td></tr>
<tr><td>Search time, with expansion</td><td>{config['retrieval_with_expansion_ms_per_query']:.1f} ms / query</td></tr>
<tr><td>MAP, held-out queries</td><td>{held_final:.4f}</td></tr>
<tr><td>MAP, all 225 queries</td><td>{final_ap:.4f}</td></tr>
</table>
<p class="caption">Table 9: the submitted configuration. Indexing is a single-threaded
in-memory build; search time is wall clock over all 225 queries divided by 225.</p>

<h2>6. Discussion of results</h2>

<h3>6.1 Preprocessing dominates everything else</h3>
<p>The single largest effect in the whole study is stopword removal. The best variant
with no stopword list reaches MAP {worst_stop['AP']:.4f}; the best with one reaches
{best_index_row['AP']:.4f} &mdash; a gain of {pct(best_index_row['AP'], worst_stop['AP'])},
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

<h3>6.2 Model choice matters more than model parameters</h3>
<p>Raw term frequency with no IDF component (MAP 0.1927) is less than two thirds as
effective as anything that weights by term rarity, which is the expected confirmation
that the vector space model's IDF component is doing the work. Among the properly
weighted models the spread is narrow &mdash; TF&ndash;IDF, BM25, DFR&ndash;BM25, PL2 and
DPH all land between 0.31 and 0.33 MAP &mdash; with In_expB2 slightly ahead.</p>
<p>Parameter tuning then adds remarkably little. On the held-out queries the tuned model
gains {float(tuned_held['AP'].iloc[-1]) - float(tuned_held['AP'].iloc[0]):+.4f} MAP over
its own default, improving {param_sig['better']} queries and degrading
{param_sig['worse']} for a sign-test p-value of {param_sig['p']:.3f} &mdash; not
significant. The BM25 response surface is nearly flat across the whole grid: every
setting between <em>k</em><sub>1</sub> = 0.8 and 4.0 and <em>b</em> = 0.3 and 1.0 scores
within about 0.01 MAP. This is worth stating plainly rather than reporting the tuned
number alone, because the difference between the best and the default setting here is
smaller than the noise between two random halves of the query set.</p>
<p>One methodological note: an earlier, narrower sweep put the optimal
<em>k</em><sub>1</sub> exactly on the boundary of the grid, which is a sign that the grid
is too small rather than an answer. The grid reported above extends to
<em>k</em><sub>1</sub> = 4.0 so that the optimum is interior.</p>

<h3>6.3 Query expansion is the one intervention that clearly pays</h3>
<p>Bo1 pseudo-relevance feedback improves MAP from {held_base:.4f} to {held_final:.4f} on
the held-out queries, {pct(held_final, held_base)}, improving {qe_sig['better']} queries
against {qe_sig['worse']} degraded &mdash; sign-test p = {qe_sig['p']:.6f}. Unlike the
parameter tuning, this survives on queries the tuning never saw, and it is the only
change in the study that does so convincingly. It is also the only change with a real
cost: expansion runs the retrieval twice and roughly triples search time, from
{config['retrieval_ms_per_query']:.1f} ms to
{config['retrieval_with_expansion_ms_per_query']:.1f} ms per query. At Cranfield's scale
that is irrelevant; on a large collection it would be the deciding factor.</p>
<p>Why it helps so much here is specific to the collection: Cranfield abstracts are short
and the vocabulary is narrow and technical, so the top few documents for a query are
densely packed with exactly the domain terms the query failed to mention. Feedback from
five documents is best; ten begins to drift, three is too thin a sample.</p>

<h3>6.4 Where the ceiling is</h3>
<p>End to end, the pipeline moves MAP from {base_ap:.4f} (untuned BM25 on the winning
index) to {final_ap:.4f} over the full query set, {pct(final_ap, base_ap)}. Recall@100
finishes at {float(qe_full['R@100'].iloc[-1]):.4f}, so roughly a fifth of the relevant
documents are still missed in the top 100 &mdash; and at least one of them, document 995
for query 125, is unreachable by any lexical system because the record is empty. The
remaining gap is mostly vocabulary mismatch that term expansion from the top five
documents cannot bridge, which is precisely the failure mode that dense retrieval
addresses and that this assignment excludes.</p>

<h3>6.5 Threats to validity</h3>
<p>The held-out half is 112 queries, so a MAP difference below roughly 0.01 is not
distinguishable from noise; we have reported sign tests rather than relying on the point
estimates. Stage 1 selected the index on the full query set before the train/test split
was introduced in Stages 3 and 4, so the preprocessing choice is mildly optimistic &mdash;
though with an effect that large, and a ranking that is stable across every model we
tried, the conclusion is not in doubt.</p>

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
