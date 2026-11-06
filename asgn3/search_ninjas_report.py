"""Generate the assignment report from the recorded experiment results."""
import argparse
import csv
import json
from pathlib import Path
from xml.sax.saxutils import escape

import reportlab
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
)

from search_ninjas_group import GROUP_DETAILS, GROUP_MEMBERS, GROUP_NAME


def generate_report(root):
    root = Path(root)
    metadata = json.loads((root / "search_ninjas_metadata.json").read_text())
    final_meta = json.loads((root / "search_ninjas_final_metadata.json").read_text())
    zone_diag = json.loads((root / "search_ninjas_zone_diagnostics.json").read_text())
    or_diag = json.loads((root / "search_ninjas_zone_or_diagnostics.json").read_text())

    def read_csv(filename):
        with (root / filename).open(newline="") as stream:
            return list(csv.DictReader(stream))

    metrics = read_csv("search_ninjas_metrics.csv")
    svm_grid = read_csv("search_ninjas_svm_grid.csv")
    coefficients = read_csv("search_ninjas_svm_weights.csv")
    large_weights = sorted(
        coefficients, key=lambda row: -abs(float(row["standardized_weight"]))
    )[:3]

    def measure(method, split, key="f1", k=10):
        return float(next(
            row[key] for row in metrics
            if row["method"] == method and row["split"] == split
            and int(row["k"]) == k
        ))

    fonts = Path(reportlab.__file__).parent / "fonts"
    pdfmetrics.registerFont(TTFont("ReportSans", str(fonts / "Vera.ttf")))
    pdfmetrics.registerFont(TTFont("ReportSansBold", str(fonts / "VeraBd.ttf")))
    styles = getSampleStyleSheet()
    normal = ParagraphStyle(
        "ReportBody", parent=styles["BodyText"], fontName="ReportSans",
        fontSize=9.2, leading=13.7, spaceAfter=8,
    )
    h1 = ParagraphStyle(
        "ReportH1", parent=styles["Heading1"], fontName="ReportSansBold",
        fontSize=14, leading=19, textColor=colors.HexColor("#18344f"),
        spaceBefore=18, spaceAfter=11, keepWithNext=True,
    )
    h2 = ParagraphStyle(
        "ReportH2", parent=styles["Heading2"], fontName="ReportSansBold",
        fontSize=10.5, leading=14, textColor=colors.HexColor("#24435d"),
        spaceBefore=12, spaceAfter=7, keepWithNext=True,
    )
    cell = ParagraphStyle(
        "ReportCell", parent=normal, fontSize=8.1, leading=11, spaceAfter=0,
    )
    heading_style = ParagraphStyle(
        "CoverTitle", parent=h1, fontSize=20, leading=26,
        alignment=TA_CENTER, spaceBefore=20, spaceAfter=9,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle", parent=normal, alignment=TA_CENTER,
        fontSize=10.5, leading=15,
    )
    story = []

    def p(message):
        story.append(Paragraph(escape(message), normal))

    def h(message, level=1):
        story.append(Paragraph(escape(message), h1 if level == 1 else h2))

    def t(headers, rows, widths=None):
        data = [[Paragraph(escape(str(x)), cell) for x in row]
                for row in [headers] + list(rows)]
        table = Table(data, colWidths=widths, hAlign="LEFT", repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e6eef4")),
            ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d2d9df")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
        story.append(Spacer(1, 0.13 * inch))

    story.append(Paragraph("Programming Assignment III", heading_style))
    story.append(Paragraph("Learning to Rank", subtitle_style))
    story.append(Paragraph("Information Retrieval | Cranfield Collection", subtitle_style))
    story.append(Spacer(1, 0.18 * inch))
    h("1. Group details")
    p(f"Group name: {GROUP_NAME}")
    t(["Student ID", "Member"], [member.split(" ", 1) for member in GROUP_MEMBERS],
      [1.35 * inch, 4.85 * inch])

    h("2. Problem statement")
    p(
        "The objective is to learn a ranking function for the Cranfield collection "
        "using relevance judgments. Method A assigns weights to Boolean matches "
        "in four document fields. Method B uses a linear support vector machine "
        "over query-document retrieval features. The methods are compared with "
        "precision, recall, and F1 at fixed rank cutoffs."
    )
    p(
        f"The collection contains {metadata['documents']} documents and "
        f"{metadata['queries']} queries. The experiment uses the supplied "
        "Cranfield relevance judgments, rather than constructing new labels. "
        "Additional query files can be ranked without relevance judgments."
    )

    h("3. Implementation details")
    h("3.1 Data representation", 2)
    p(
        "Each document is split into title, author, bibliography, and body. "
        "The parser retains empty fields and concatenates repeated body sections. "
        "Text is converted to lowercase alphanumeric tokens and the scikit-learn "
        "English stopword list is removed; no stemming is applied. Document "
        "statistics are computed on the complete collection, which is known "
        "at retrieval time."
    )
    p(
        "Cranfield query labels are non-contiguous, whereas the judgment file "
        "uses sequential query positions. The implementation records both "
        "identifiers. Judgment codes 1-4 are positive; code -1 and unjudged "
        "pairs are treated as nonrelevant. This is a binary relevance setting "
        "with incomplete judgments, not a claim that all unjudged pairs are "
        "truly irrelevant."
    )
    story.append(PageBreak())

    h("3.2 Method A: Boolean weighted zones", 2)
    p(
        "For a field f, its zone value is the number of distinct query terms "
        "present in f divided by the number of distinct query terms. The value "
        "is zero for an empty query. The document score is the sum of these "
        "four field values multiplied by their learned nonnegative weights. "
        "Thus each query term contributes a Boolean presence indicator even "
        "when it occurs repeatedly in a field."
    )
    p(
        "Weights sum to one. An exhaustive search considers 286 combinations "
        "in increments of 0.1, selecting the highest mean training F1@10. "
        "For comparisons, zone_or uses a single any-term match indicator per "
        "field, and equal_zone uses equal field weights. Each model is trained "
        "independently where applicable."
    )
    p(
        "Documents are ranked by the weighted field scores. When scores tie, "
        "the implementation uses increasing numeric document ID. "
        "The same ordering rule is applied in evaluation and prediction."
    )
    t(
        ["Field", "Learned per-term weight", "OR baseline weight"],
        [[name, f"{value:.1f}", f"{metadata['zone_or_weights'][name]:.1f}"]
         for name, value in metadata["zone_weights"].items()],
        [1.5 * inch, 2.35 * inch, 2.35 * inch],
    )

    h("3.3 Method B: linear SVM", 2)
    p(
        "The feature vector contains 22 values per document field, for 88 "
        "features in total. StandardScaler is fitted on training query-document "
        "pairs only. An SGDClassifier with hinge loss, L2 regularization, "
        "balanced class weights, and coefficient averaging fits a linear "
        "decision function. All training query-document pairs are used; "
        "there is no negative sampling. Documents are ranked by their signed "
        "decision values, without applying a classification threshold."
    )
    t(
        ["Feature family", "Features computed for each of the four fields"],
        [
            ["Query coverage", "Number of matched distinct terms and matched-term ratio"],
            ["Field length and IDF", "Token count and sum of IDF for matched terms"],
            ["Raw term frequency", "Sum, minimum, maximum, mean, and variance"],
            ["Length-normalized TF", "Sum, minimum, maximum, mean, and variance"],
            ["TF-IDF", "Sum, minimum, maximum, mean, and variance"],
            ["Retrieval scores", "Boolean OR match, vector-space cosine, and BM25"],
        ],
        [1.75 * inch, 4.45 * inch],
    )
    p(
        "Out-of-vocabulary query terms contribute zero term frequency. "
        "Empty fields have zero matching, cosine, and BM25 features. "
        "The 88-feature definitions are unchanged between the original "
        "and revised zone experiments."
    )
    story.append(PageBreak())

    h("4. Plan of experiments")
    p(
        "A seeded query-level split assigns 135 queries to training, 45 to "
        "validation, and 45 to test. All candidate documents for a query stay "
        "in the same partition. Method A weights are chosen on training F1@10. "
        "Method B's regularization parameter is chosen on validation F1@10 "
        "from 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, and 100. "
        "The evaluation SVM is retained without refitting."
    )
    p(
        "For each query, precision is relevant documents retrieved divided "
        "by k, recall is relevant documents retrieved divided by all known "
        "relevant documents, and F1 is their harmonic mean. Each reported "
        "value is the arithmetic mean of per-query values. Cutoffs are "
        "5, 10, 20, 50, and 100, with F1@10 used for selection."
    )
    p(
        "The body-only BM25 and equal-zone rankings are untuned references. "
        "A separate final model is fitted to the 180 training and validation "
        "queries for scoring unknown queries. The final model does not "
        "contribute to the reported test metrics."
    )
    t(
        ["SVM alpha", "Validation F1@10", "Epochs"],
        [[row["alpha"], f"{float(row['validation_f1_at_10']):.4f}", row["epochs"]]
         for row in svm_grid],
        [1.6 * inch, 2.25 * inch, 2.35 * inch],
    )
    p(
        f"Selected alpha: {metadata['svm_alpha']:g}. The evaluation scaler "
        f"was fitted on {metadata['train_pairs']:,} pairs, including "
        f"{metadata['train_positive_pairs']:,} positive labels."
    )
    story.append(PageBreak())

    h("5. Results")
    h("5.1 Performance at rank 10", 2)
    t(
        ["Method", "Split", "P@10", "R@10", "F1@10"],
        [[row["method"], row["split"],
          f"{float(row['precision']):.4f}",
          f"{float(row['recall']):.4f}",
          f"{float(row['f1']):.4f}"]
         for row in metrics if int(row["k"]) == 10 and row["split"] in ("validation", "test")],
        [1.5 * inch, 1.25 * inch, 1.15 * inch, 1.15 * inch, 1.15 * inch],
    )
    h("5.2 Test performance across cutoffs", 2)
    t(
        ["Method", "k", "Precision", "Recall", "F1"],
        [[row["method"], row["k"],
          f"{float(row['precision']):.4f}",
          f"{float(row['recall']):.4f}",
          f"{float(row['f1']):.4f}"]
         for row in metrics if row["split"] == "test"],
        [1.5 * inch, 0.7 * inch, 1.3 * inch, 1.3 * inch, 1.4 * inch],
    )
    h("5.3 Model diagnostics", 2)
    t(
        ["Zone variant", "Best weight tuples", "Mean distinct scores/query", "First test top-score ties"],
        [[diag["method"], diag["best_weight_tuples"],
          f"{diag['mean_distinct_scores']:.1f}", diag["first_test_top_ties"]]
         for diag in (zone_diag, or_diag)],
        [1.1 * inch, 1.55 * inch, 1.85 * inch, 1.7 * inch],
    )
    h("6. Discussion of results")
    p(
        f"The original OR-based zone method has low score resolution. "
        f"At least one query term appears in {or_diag['body_match_fraction']:.1%} "
        f"of document bodies, but its selected weighted score takes only "
        f"{or_diag['mean_distinct_scores']:.1f} distinct values per query on average. "
        f"For the first test query, {or_diag['first_test_top_ties']} documents "
        "share the top score. Most of their relative order is therefore "
        "determined by the document-ID tie rule."
    )
    p(
        f"Matching individual query terms gives {zone_diag['mean_distinct_scores']:.1f} "
        f"distinct scores per query on average. On the test split, F1@10 "
        f"increases from {measure('zone_or', 'test'):.4f} for OR zones to "
        f"{measure('zone', 'test'):.4f} for per-term zones. This gain comes "
        "from representing partial query coverage, not from adding term "
        "frequency or BM25 to Method A."
    )
    p(
        f"Equal-weight zones achieve test F1@10 of "
        f"{measure('equal_zone', 'test'):.4f}, compared with "
        f"{measure('zone', 'test'):.4f} for the learned zone weights. "
        "The learned weights do not consistently improve this baseline "
        "on the test split; selection optimizes the training partition only."
    )
    p(
        f"On validation, the SVM scores {measure('svm', 'validation'):.4f} "
        f"F1@10 and body BM25 scores {measure('body_bm25', 'validation'):.4f}. "
        f"On test, their respective values are {measure('svm', 'test'):.4f} "
        f"and {measure('body_bm25', 'test'):.4f}. The test difference of "
        f"{measure('svm', 'test') - measure('body_bm25', 'test'):+.4f} "
        "is small. The coefficients give weight to several lexical "
        "retrieval signals, but correlated features make coefficient "
        "magnitude an imperfect indication of individual feature importance."
    )
    top_coefficients = ", ".join(
        f"{row['feature']} ({float(row['standardized_weight']):+.4f})"
        for row in large_weights
    )
    p(f"The three largest absolute standardized SVM coefficients are {top_coefficients}.")
    p(
        "There are important limits to interpretation. Only one split of a "
        "relatively small query collection is reported, and incomplete "
        "judgments are counted as negatives. The SVM optimizes pointwise "
        "hinge loss rather than ranked-list F1. Furthermore, the expanded "
        "regularization grid and per-term zone formulation were introduced "
        "after earlier test results had been inspected, so this is not a "
        "freshly untouched test evaluation. Larger independent query sets "
        "or repeated query-level cross-validation would give stronger "
        "evidence about generalization."
    )
    p(
        f"For unknown queries, the final model refits the SVM and zone weights "
        f"on {len(final_meta['fit_qids'])} training and validation queries. "
        f"The {len(final_meta['excluded_test_qids'])} test queries remain "
        "excluded from fitting. The executable writes positional-ID and "
        "original-label TREC runs with the model scores and recorded ranks."
    )
    h("References", 2)
    p(
        "Programming Assignment III: Learning to Rank (course handout); "
        "Cranfield collection and relevance judgments (supplied files); "
        "scikit-learn documentation for StandardScaler and SGDClassifier; "
        "ir_datasets Cranfield relevance-code definitions."
    )

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("ReportSans", 8)
        canvas.setFillColor(colors.HexColor("#607184"))
        canvas.drawString(0.72 * inch, 0.4 * inch, GROUP_NAME)
        canvas.drawRightString(7.5 * inch, 0.4 * inch, f"{doc.page}")
        canvas.restoreState()

    SimpleDocTemplate(
        str(root / "search_ninjas_report.pdf"),
        pagesize=(595.276, 841.89),
        title="Programming Assignment III: Learning to Rank",
        author=GROUP_DETAILS,
        leftMargin=0.72 * inch, rightMargin=0.72 * inch,
        topMargin=0.58 * inch, bottomMargin=0.65 * inch,
    ).build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts", type=Path,
        default=Path(__file__).resolve().parent / "search_ninjas_artifacts",
    )
    generate_report(parser.parse_args().artifacts)
