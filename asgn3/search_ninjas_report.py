"""Generate an experiment report directly from the recorded run artifacts."""
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape
import reportlab
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from search_ninjas_group import GROUP_DETAILS


def generate_report(root):
    root = Path(root)
    metadata = json.loads((root / 'search_ninjas_metadata.json').read_text())
    def read(name):
        with (root / name).open() as handle:
            return list(csv.DictReader(handle))
    metrics = read('search_ninjas_metrics.csv')
    styles = getSampleStyleSheet()
    fonts = Path(reportlab.__file__).parent / 'fonts'
    pdfmetrics.registerFont(TTFont('ReportSans', str(fonts / 'Vera.ttf')))
    pdfmetrics.registerFont(TTFont('ReportSansBold', str(fonts / 'VeraBd.ttf')))
    styles['BodyText'].fontName = 'ReportSans'
    styles['BodyText'].fontSize = 9
    styles['BodyText'].leading = 13
    styles['Heading1'].fontName = 'ReportSansBold'
    styles['Heading1'].fontSize = 17
    styles['Heading1'].leading = 22
    story = []
    def heading(text):
        story.append(Paragraph(escape(text), styles['Heading1']))
    def paragraph(text):
        story.append(Paragraph(escape(text), styles['BodyText']))
        story.append(Spacer(1, .12 * inch))
    def table(headers, rows):
        data = [[Paragraph(escape(str(c)), styles['BodyText']) for c in row]
                for row in [headers] + rows]
        item = Table(data, repeatRows=1, hAlign='LEFT')
        item.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#dbe9f4')),
            ('GRID', (0, 0), (-1, -1), .3, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
        story.append(item)
    def page():
        story.append(PageBreak())
    heading('Programming Assignment III: Learning to Rank')
    heading('1. Group details')
    paragraph('Group: ' + GROUP_DETAILS)
    paragraph('This report is generated from a completed experiment.')
    heading('2. Problem statement')
    paragraph('The task is ranked retrieval on Cranfield using (A) learned Boolean zone weights and (B) a linear support vector machine. Both return an ordered document list. Precision, recall, and F1 measure binary relevance at fixed cutoffs.')
    paragraph(f"The collection contains {metadata['documents']} documents and {metadata['queries']} queries. The deterministic split uses seed {metadata['seed']}: {metadata['split_sizes']}. Model fitting uses training labels and alpha selection uses validation labels. This revised study follows inspection of earlier test results.")
    paragraph('The SVM is a pointwise binary relevance classifier whose signed decision function is used as a ranking score. It is not a pairwise RankSVM. This interpretation follows the brief’s instruction to learn linear weights from query-document features and relevance judgments.')
    page()
    heading('3. Implementation details')
    heading('3.1 Data and preprocessing')
    paragraph('Documents are parsed into title, author, bibliography, and body. Repeated body sections are appended. Stray .A and .B tags encountered inside the body do not discard the remaining body text. Empty documents remain in the corpus and candidate set.')
    paragraph('Text is lowercased and split into alphanumeric tokens. The scikit-learn English stopword list is applied identically to documents and queries. No stemming is used. This fixed preprocessing is chosen before evaluation rather than tuned against the test split.')
    paragraph('Cranfield query labels are non-contiguous. Relevance judgments use query positions 1 through 225, not the original .I labels. Training and evaluation therefore use position IDs. Each output is also exported using original labels for evaluation systems that expect those labels.')
    paragraph('Judgment codes 1 through 4 are binary relevant. Code -1 is treated as nonrelevant in both SVM training and evaluation, as are code 5 and absent judgments. There are 1,837 judgment rows: 1,612 positive and 225 coded -1. The ir_datasets Cranfield implementation defines -1 as “References of no interest”; the supplied readme does not explicitly define -1. High lexical similarity does not establish a positive relevance judgment. Treating unjudged pairs as negatives is a conventional closed-collection assumption, not proof that every unjudged document is irrelevant. The reported precision may underestimate relevance when judgments are incomplete.')
    paragraph('Document frequency and vocabulary are computed on the entire document collection, which is available at retrieval time. Only query relevance labels are held out. This is query generalization within a fixed collection, not a claim about unseen document collections.')
    table(['Input', 'SHA-256'], [[name, digest] for name, digest in metadata['inputs_sha256'].items()])
    page()
    heading('3.2 Method A: Boolean weighted zones')
    paragraph('Method A (zone) averages per-term Boolean presence within each field: z_f(q,d) = number of distinct query terms present in field f / number of distinct query terms. An empty query gives zero. The final score is the weighted sum of these field fractions, equivalently averaging Boolean weighted zone scores over query terms. This is a per-term extension of Boolean zone scoring, not a binary whole-field indicator. The original any-term OR method is retained as zone_or with separately learned weights. All documents are scored; exact ties use ascending numeric document ID.')
    paragraph('The four nonnegative weights sum to one. For each of zone and zone_or separately, an exhaustive simplex grid with increments of 0.1 evaluates 286 weight combinations. Mean training F1@10 selects the winner; exact objective ties keep the first grid combination in lexicographic title/author/bibliography order. Validation and test labels do not select these weights.')
    table(['Field', 'Per-term weight', 'OR baseline weight'], [[k, f'{v:.1f}', f"{metadata['zone_or_weights'][k]:.1f}"] for k,v in metadata['zone_weights'].items()])
    grid = sorted(read('search_ninjas_zone_grid.csv'), key=lambda r: -float(r['train_f1_at_10']))[:8]
    table(['Title','Author','Bibliography','Body','Training F1@10'],
          [[r[x] for x in ('title','author','bibliography','body')] +
           [f"{float(r['train_f1_at_10']):.4f}"] for r in grid])
    diagnostics = json.loads((root / 'search_ninjas_zone_diagnostics.json').read_text())
    or_diagnostics = json.loads((root / 'search_ninjas_zone_or_diagnostics.json').read_text())
    table(['Method', 'Best weight tuples', 'Mean distinct scores', 'First test top ties'],
          [[d['method'], d['best_weight_tuples'], f"{d['mean_distinct_scores']:.2f}", d['first_test_top_ties']]
           for d in (diagnostics, or_diagnostics)])
    paragraph('The OR baseline can represent at most 16 field-presence patterns. Identical patterns tie regardless of grid resolution. Per-term fractions distinguish partial query coverage without introducing term frequency or BM25 into Method A. Equal_zone assigns equal weights to the new field fractions. The 88 SVM features retain their original definitions, including the Boolean OR feature.')
    page()
    heading('3.3 Method B: feature definitions')
    paragraph('The same 22 features are computed for each field, giving 88 features per query-document pair. Statistics operate over distinct preprocessed query terms, including zero frequencies for absent and out-of-vocabulary terms. Query-term multiplicity affects the cosine query vector, while BM25 sums once per distinct term.')
    table(['Features per field', 'Definition'], [
        ['Query term number / ratio', 'Number of distinct query terms present; divided by the number of distinct query terms (zero for an empty query).'],
        ['Stream length', 'Number of preprocessed tokens in the field.'],
        ['IDF', 'Sum of IDF over matched distinct query terms; IDF(t)=1+ln((N+1)/(df(t)+1)).'],
        ['Term frequency: 5 statistics', 'Sum, minimum, maximum, arithmetic mean, and population variance of raw counts over distinct query terms.'],
        ['Normalized TF: 5 statistics', 'The same five statistics after dividing counts by max(field length,1).'],
        ['TF-IDF: 5 statistics', 'The same five statistics of raw frequency multiplied by smoothed field-specific IDF.'],
        ['Boolean model', '1 if any query term occurs in the field, otherwise 0.'],
        ['Vector space model', 'Cosine similarity of L2-normalized (1+ln(tf))*IDF vectors; zero for empty vectors. Document norms use the complete field vocabulary.'],
        ['BM25', 'Sum ln(1+(N-df+0.5)/(df+0.5))*tf*2.2 / (tf+1.2*(0.25+0.75*length/average_length)).']])
    paragraph('Empty fields produce zero match, frequency, cosine and BM25 features. Out-of-vocabulary terms contribute zero TF and remain in statistic denominators. No label-derived feature is present.')
    page()
    heading('3.4 SVM training and selection')
    paragraph(f"All {metadata['train_pairs']} training query-document pairs are used, including {metadata['train_positive_pairs']} relevant pairs. No negative subsampling is performed. StandardScaler learns mean and scale only from training pairs. SGDClassifier uses balanced class weights to compensate for relevance imbalance.")
    paragraph('The linear hinge-loss SVM uses an L2 penalty and stochastic gradient optimization with averaged coefficients. Training shuffles pairs reproducibly using the split seed. It stops after 10 epochs without training-loss improvement of 1e-4, with a maximum of 2,000 epochs. A convergence warning fails the experiment. Numerical library threads are limited to two. This is a scalable approximate optimizer of the linear SVM objective, not an exact quadratic-program solver.')
    paragraph('Regularization alpha is selected from 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, and 100 by mean validation F1@10. Ties keep the first candidate (smaller alpha). The selected training-only model is retained without refitting on validation queries. This keeps the reported validation result and saved model consistent.')
    table(['Alpha', 'Validation F1@10'], [[r['alpha'], f"{float(r['validation_f1_at_10']):.4f}"] for r in read('search_ninjas_svm_grid.csv')])
    paragraph(f"Selected alpha: {metadata['svm_alpha']} ({metadata['svm_epochs']} epochs). Ranking score is w·standardize(x)+b. The intercept is {metadata['svm_intercept']:.6f}. Scores are not relevance probabilities, and negative scores are still ranked. Fixed-cutoff evaluation does not apply a zero-score threshold.")
    weights = sorted(read('search_ninjas_svm_weights.csv'), key=lambda r: -abs(float(r['standardized_weight'])))[:12]
    table(['Largest absolute standardized weights', 'Weight'], [[r['feature'], f"{float(r['standardized_weight']):.5f}"] for r in weights])
    paragraph('Weights measure associations conditional on correlated features; they are not causal feature importance. All 88 standardized and raw-scale coefficients are exported as CSV.')
    page()
    heading('4. Plan of experiments')
    paragraph('A seeded permutation partitions queries into 60% train, 20% validation, and 20% test. All document pairs belonging to a query remain together. The split JSON records both positional IDs and original labels. Within each run, alpha is selected on validation F1 before test metrics are computed. However, the expanded grid and per-term zone revision follow inspection of previous test results; the reported test evaluation is not a fresh untouched experiment.')
    paragraph('For a query, let h be the number of relevant documents among the top k and r the total number of judged relevant documents. Precision=h/k; recall=h/r (zero if r=0); F1=2PR/(P+R), or zero when P+R=0. Results are macro averages over queries. Mean F1 is the average of per-query F1 values, not the harmonic mean of macro precision and macro recall.')
    paragraph('Cutoffs are 5, 10, 20, 50 and 100. F1@10 is fixed as the primary selection metric. Equal zone weighting and body-only BM25 provide untuned baselines. Neither baseline is selected using test results.')
    page()
    heading('5. Results')
    table(['Method', 'Split', 'P@10', 'R@10', 'F1@10'],
          [[r['method'],r['split']] + [f"{float(r[x]):.4f}" for x in ('precision','recall','f1')]
           for r in metrics if r['k']=='10'])
    paragraph(metadata['revision_note'])
    paragraph('No significance claim is made from this single split. A future study could use nested cross-validation across queries, but that is separate from the fixed evaluation reported here.')
    page()
    heading('5.1 Results across cutoffs')
    table(['Method','k','Precision','Recall','F1'],
          [[r['method'],r['k']] + [f"{float(r[x]):.4f}" for x in ('precision','recall','f1')]
           for r in metrics if r['split']=='test'])
    test10 = {r['method']:float(r['f1']) for r in metrics if r['split']=='test' and r['k']=='10'}
    paragraph(f"At the primary cutoff, SVM minus learned-zone F1 is {test10['svm']-test10['zone']:+.4f}. Learned-zone minus equal-zone F1 is {test10['zone']-test10['equal_zone']:+.4f}. SVM minus body-BM25 F1 is {test10['svm']-test10['body_bm25']:+.4f}. These are observed differences on this held-out split, not guaranteed improvements on future queries.")
    page()
    heading('6. Discussion of results')
    def f1(method, partition):
        return float(next(r['f1'] for r in metrics if r['method'] == method and r['split'] == partition and r['k'] == '10'))
    paragraph(f"The OR baseline collapses much of query coverage into a single match bit: {or_diagnostics['body_match_fraction']:.1%} of document body zones match at least one query term. It yields {or_diagnostics['mean_distinct_scores']:.2f} distinct scores per query on average, with {or_diagnostics['first_test_top_ties']} documents tied at the top for test query {or_diagnostics['first_test_qid']}, and {or_diagnostics['best_weight_tuples']} weight tuples sharing the best training objective. Document-ID tie-breaking then determines much of the order within those groups.")
    paragraph(f"Per-term presence changes mean distinct scores to {diagnostics['mean_distinct_scores']:.2f}, top-score ties on the same query to {diagnostics['first_test_top_ties']}, and best weight tuples to {diagnostics['best_weight_tuples']}. Test F1@10 changes from {f1('zone_or', 'test'):.4f} (OR) to {f1('zone', 'test'):.4f} (per-term), a difference of {f1('zone', 'test')-f1('zone_or', 'test'):+.4f}. This reflects increased query-coverage resolution; repeated occurrences still do not increase a term's Boolean contribution.")
    for partition in ('validation', 'test'):
        svm_f1, bm25_f1 = f1('svm', partition), f1('body_bm25', partition)
        paragraph(f"On {partition}, SVM F1@10 is {svm_f1:.4f} and body BM25 is {bm25_f1:.4f}, a difference of {svm_f1-bm25_f1:+.4f}. These are observed macro-average differences on this split, not proof of statistical superiority.")
    leaders = ', '.join(f"{w['feature']} ({float(w['standardized_weight']):+.4f})" for w in weights[:3])
    paragraph(f"Selected alpha is {metadata['svm_alpha']:g}, compared with the smallest tested value {min(metadata['alpha_grid']):g}; larger alpha means stronger L2 regularization. The largest absolute standardized coefficients are {leaders}. This is consistent with a regularized combination of lexical signals. Correlated features prevent interpreting coefficient magnitudes as independent importance, and the experiment does not establish a clear advantage over BM25 across future queries.")
    for partition in ('validation', 'test'):
        learned, equal = f1('zone', partition), f1('equal_zone', partition)
        paragraph(f"On {partition}, learned per-term zone weights give F1@10 {learned:.4f}, versus {equal:.4f} with equal weights (difference {learned-equal:+.4f}). Weight tuning is assessed separately from the larger change in the matching definition; a difference on this split is not a guaranteed generalization gain.")
    paragraph(f"Limitations include a single split of only {metadata['queries']} queries, incomplete judgments (unjudged pairs count as negatives), and a pointwise classification loss rather than optimization of ranked-list F1. The zone change was motivated after earlier test inspection, so model development has indirectly been informed by this test set. Additional independent queries or nested cross-validation would be needed for stronger generalization claims.")
    paragraph('Empty documents cannot obtain lexical evidence even when judged relevant. Queries reduced to no known terms receive deterministic rankings with little or no evidence; these are not silently omitted. The fixed corpus and collection statistics are stored with the model so unseen query files use exactly the training feature definitions.')
    paragraph(f"Recorded feature extraction time: {metadata['feature_seconds']:.2f} seconds. Recorded evaluation-experiment time before final-model refitting, serialization and PDF generation: {metadata['total_seconds']:.2f} seconds. Timings depend on the machine and exclude dependency installation.")
    paragraph('Reproduce from asgn3: python3 -m venv .venv; .venv/bin/pip install -r requirements.txt; .venv/bin/python -m unittest discover -s tests -v; .venv/bin/python search_ninjas_rank.py train.')
    paragraph('For additional queries: .venv/bin/python search_ninjas_rank.py predict --queries your_queries.qry. The query file uses .I labels and .W text. Prediction does not load relevance judgments. Original-label and position-ID outputs are written separately in six-column TREC format: qid Q0 docno rank score search_ninjas.')
    final = json.loads((root / 'search_ninjas_final_metadata.json').read_text())
    paragraph(f"Separate hidden-query deployment model: search_ninjas_final_model.joblib refits the scaler and SVM on {len(final['fit_qids'])} train+validation queries at the selected alpha, and relearns pure zone weights on those same queries. The {len(final['excluded_test_qids'])} test queries are excluded. No reported metrics belong to this model. The original search_ninjas_model.joblib remains the evaluation model. Prediction defaults to the final model; --model selects either explicitly. Final model provenance is in search_ninjas_final_metadata.json.")
    paragraph('Deliverables include the Python programs, dependency pins, split and configuration JSON, weight and metric tables, two held-out result lists per model, a serialized model, and this PDF. Artifacts are generated under search_ninjas_artifacts and are tracked by Git. The packaging script includes source, inputs, tests, models, results and this report, while excluding the virtual environment. Load only trusted joblib model files because their deserialization can execute code.')
    paragraph('References: supplied PA3 assignment brief; supplied cranqrel.readme for relevance codes; scikit-learn documentation for StandardScaler and SGDClassifier; the standard BM25 weighting formula. Relevance-label convention: https://github.com/allenai/ir_datasets/blob/master/ir_datasets/datasets/cranfield.py (QREL_DEFS, accessed 6 November 2026). No earlier-assignment code is imported.')
    def footer(canvas, doc):
        canvas.setFont('ReportSans', 9)
        canvas.drawRightString(7.5*inch, .4*inch, f'search_ninjas | {doc.page}')
    SimpleDocTemplate(str(root / 'search_ninjas_report.pdf'),
                      title='Programming Assignment III: Learning to Rank',
                      author=GROUP_DETAILS,
                      topMargin=.6*inch, bottomMargin=.6*inch).build(
                          story, onFirstPage=footer, onLaterPages=footer)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path,
                        default=Path(__file__).resolve().parent / 'search_ninjas_artifacts')
    generate_report(parser.parse_args().artifacts)
