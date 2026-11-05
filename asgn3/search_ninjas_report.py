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
    paragraph('Group: ' + metadata['group'])
    paragraph('This report is generated from a completed experiment. Group details are recorded for this run; any missing student IDs should be supplied before submission.')
    paragraph('The task is ranked retrieval on Cranfield using (A) learned Boolean zone weights and (B) a linear support vector machine. Both return an ordered document list. Precision, recall, and F1 measure binary relevance at fixed cutoffs.')
    paragraph(f"The collection contains {metadata['documents']} documents and {metadata['queries']} queries. The deterministic split uses seed {metadata['seed']}: {metadata['split_sizes']}. No test labels are used to choose weights, preprocessing, or hyperparameters.")
    paragraph('The SVM is a pointwise binary relevance classifier whose signed decision function is used as a ranking score. It is not a pairwise RankSVM. This interpretation follows the brief’s instruction to learn linear weights from query-document features and relevance judgments.')
    table(['Method', 'Test P@10', 'Test R@10', 'Test F1@10'],
          [[r['method']] + [f"{float(r[x]):.4f}" for x in ('precision','recall','f1')]
           for r in metrics if r['split']=='test' and r['k']=='10'])
    page()
    heading('1. Data and preprocessing')
    paragraph('Documents are parsed into title, author, bibliography, and body. Repeated body sections are appended. Stray .A and .B tags encountered inside the body do not discard the remaining body text. Empty documents remain in the corpus and candidate set.')
    paragraph('Text is lowercased and split into alphanumeric tokens. The scikit-learn English stopword list is applied identically to documents and queries. No stemming is used. This fixed preprocessing is chosen before evaluation rather than tuned against the test split.')
    paragraph('Cranfield query labels are non-contiguous. Relevance judgments use query positions 1 through 225, not the original .I labels. Training and evaluation therefore use position IDs. Each output is also exported using original labels for evaluation systems that expect those labels.')
    paragraph('Judgment codes 1 through 4 are binary relevant, and -1 markers are ignored. Code 5 and absent judgments count as nonrelevant. Treating unjudged pairs as negatives is a conventional closed-collection assumption, not proof that every unjudged document is irrelevant. The reported precision may underestimate relevance when judgments are incomplete.')
    paragraph('Document frequency and vocabulary are computed on the entire document collection, which is available at retrieval time. Only query relevance labels are held out. This is query generalization within a fixed collection, not a claim about unseen document collections.')
    table(['Input', 'SHA-256'], [[name, digest] for name, digest in metadata['inputs_sha256'].items()])
    page()
    heading('2. Method A: Boolean weighted zones')
    paragraph('For each query and document, a zone is 1 if at least one preprocessed query term occurs in that field and 0 otherwise (OR matching). The score is the weighted sum of the four zone values. All documents are scored. Ties are resolved by ascending numeric document ID.')
    paragraph('The four nonnegative weights sum to one. An exhaustive simplex grid with increments of 0.1 evaluates 286 weight combinations. Mean training F1@10 selects the winner; exact objective ties keep the first grid combination in lexicographic title/author/bibliography order. Validation and test labels do not select these weights.')
    table(['Field', 'Learned weight'], [[k, f'{v:.1f}'] for k,v in metadata['zone_weights'].items()])
    grid = sorted(read('search_ninjas_zone_grid.csv'), key=lambda r: -float(r['train_f1_at_10']))[:8]
    table(['Title','Author','Bibliography','Body','Training F1@10'],
          [[r[x] for x in ('title','author','bibliography','body')] +
           [f"{float(r['train_f1_at_10']):.4f}"] for r in grid])
    paragraph('The model has only 16 possible Boolean zone patterns. Many documents consequently tie even with learned weights. Numeric-ID tie breaking is deterministic but not relevance-aware. This limits the model’s ranking resolution and motivates real-valued features.')
    page()
    heading('3. Method B: feature definitions')
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
    heading('4. SVM training and selection')
    paragraph(f"All {metadata['train_pairs']} training query-document pairs are used, including {metadata['train_positive_pairs']} relevant pairs. No negative subsampling is performed. StandardScaler learns mean and scale only from training pairs. SGDClassifier uses balanced class weights to compensate for relevance imbalance.")
    paragraph('The linear hinge-loss SVM uses an L2 penalty and stochastic gradient optimization with averaged coefficients. Training shuffles pairs reproducibly using the split seed. It stops after 10 epochs without training-loss improvement of 1e-4, with a maximum of 2,000 epochs. A convergence warning fails the experiment. Numerical library threads are limited to two. This is a scalable approximate optimizer of the linear SVM objective, not an exact quadratic-program solver.')
    paragraph('Regularization alpha is selected from 0.00001, 0.0001, 0.001, and 0.01 by mean validation F1@10. Ties keep the first candidate (smaller alpha). The selected training-only model is retained without refitting on validation queries. This keeps the reported validation result and saved model consistent.')
    table(['Alpha', 'Validation F1@10'], [[r['alpha'], f"{float(r['validation_f1_at_10']):.4f}"] for r in read('search_ninjas_svm_grid.csv')])
    paragraph(f"Selected alpha: {metadata['svm_alpha']} ({metadata['svm_epochs']} epochs). Ranking score is w·standardize(x)+b. The intercept is {metadata['svm_intercept']:.6f}. Scores are not relevance probabilities, and negative scores are still ranked. Fixed-cutoff evaluation does not apply a zero-score threshold.")
    weights = sorted(read('search_ninjas_svm_weights.csv'), key=lambda r: -abs(float(r['standardized_weight'])))[:12]
    table(['Largest absolute standardized weights', 'Weight'], [[r['feature'], f"{float(r['standardized_weight']):.5f}"] for r in weights])
    paragraph('Weights measure associations conditional on correlated features; they are not causal feature importance. All 88 standardized and raw-scale coefficients are exported as CSV.')
    page()
    heading('5. Experiment design and evaluation')
    paragraph('A seeded permutation partitions queries into 60% train, 20% validation, and 20% test. All document pairs belonging to a query remain together. The split JSON records both positional IDs and original labels. Test metrics are computed only after model selection.')
    paragraph('For a query, let h be the number of relevant documents among the top k and r the total number of judged relevant documents. Precision=h/k; recall=h/r (zero if r=0); F1=2PR/(P+R), or zero when P+R=0. Results are macro averages over queries. Mean F1 is the average of per-query F1 values, not the harmonic mean of macro precision and macro recall.')
    paragraph('Cutoffs are 5, 10, 20, 50 and 100. F1@10 is fixed as the primary selection metric. Equal zone weighting and body-only BM25 provide untuned baselines. Neither baseline is selected using test results.')
    table(['Method', 'Split', 'P@10', 'R@10', 'F1@10'],
          [[r['method'],r['split']] + [f"{float(r[x]):.4f}" for x in ('precision','recall','f1')]
           for r in metrics if r['k']=='10'])
    paragraph('No significance claim is made from this single split. A future study could use nested cross-validation across queries, but that is separate from the fixed evaluation reported here.')
    page()
    heading('6. Held-out results across cutoffs')
    table(['Method','k','Precision','Recall','F1'],
          [[r['method'],r['k']] + [f"{float(r[x]):.4f}" for x in ('precision','recall','f1')]
           for r in metrics if r['split']=='test'])
    test10 = {r['method']:float(r['f1']) for r in metrics if r['split']=='test' and r['k']=='10'}
    paragraph(f"At the primary cutoff, SVM minus learned-zone F1 is {test10['svm']-test10['zone']:+.4f}. Learned-zone minus equal-zone F1 is {test10['zone']-test10['equal_zone']:+.4f}. SVM minus body-BM25 F1 is {test10['svm']-test10['body_bm25']:+.4f}. These are observed differences on this held-out split, not guaranteed improvements on future queries.")
    page()
    heading('7. Discussion, limitations, and reproducibility')
    paragraph('Zone scoring discards term frequency and document length, so many ties are inevitable. The SVM can use richer features but optimizes a classification loss rather than the ranked-list F1 objective. Correlated features, incomplete relevance judgments, a small query set, and the chosen preprocessing constrain generalization. A learned model need not outperform BM25.')
    paragraph('Empty documents cannot obtain lexical evidence even when judged relevant. Queries reduced to no known terms receive deterministic rankings with little or no evidence; these are not silently omitted. The fixed corpus and collection statistics are stored with the model so unseen query files use exactly the training feature definitions.')
    paragraph(f"Recorded feature extraction time: {metadata['feature_seconds']:.2f} seconds. Recorded training/evaluation time before serialization and PDF generation: {metadata['total_seconds']:.2f} seconds. Timings depend on the machine and exclude dependency installation.")
    paragraph('Reproduce from asgn3: python3 -m venv .venv; .venv/bin/pip install -r requirements.txt; .venv/bin/python -m unittest discover -s tests -v; .venv/bin/python search_ninjas_rank.py train. Supply --group with actual member names and IDs for the submission report.')
    paragraph('For additional queries: .venv/bin/python search_ninjas_rank.py predict --queries your_queries.qry. The query file uses .I labels and .W text. Prediction does not load relevance judgments. Original-label and position-ID outputs are written separately in six-column TREC format: qid Q0 docno rank score search_ninjas.')
    paragraph('Deliverables include the Python programs, dependency pins, split and configuration JSON, weight and metric tables, two held-out result lists per model, a serialized model, and this PDF. Artifacts are generated under search_ninjas_artifacts, which is ignored by Git. Archive that directory explicitly when preparing a submission. Load only trusted joblib model files because their deserialization can execute code.')
    paragraph('References: supplied PA3 assignment brief; supplied cranqrel.readme for relevance codes; scikit-learn documentation for StandardScaler and SGDClassifier; the standard BM25 weighting formula. No earlier-assignment code is imported.')
    def footer(canvas, doc):
        canvas.setFont('ReportSans', 9)
        canvas.drawRightString(7.5*inch, .4*inch, f'search_ninjas | {doc.page}')
    SimpleDocTemplate(str(root / 'search_ninjas_report.pdf'),
                      title='Programming Assignment III: Learning to Rank',
                      author=metadata['group'],
                      topMargin=.6*inch, bottomMargin=.6*inch).build(
                          story, onFirstPage=footer, onLaterPages=footer)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path,
                        default=Path(__file__).resolve().parent / 'search_ninjas_artifacts')
    generate_report(parser.parse_args().artifacts)
