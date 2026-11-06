"""Train, evaluate, and apply PA3 zone and linear-SVM ranking models."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from threadpoolctl import threadpool_limits

from search_ninjas_data import FIELDS, read_documents, read_queries, read_qrels
from search_ninjas_features import Collection, feature_names
from search_ninjas_group import GROUP_DETAILS

BASE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE / 'search_ninjas_artifacts'
CUTOFFS = (5, 10, 20, 50, 100)
ALPHAS = (.00001, .0001, .001, .01, .1, 1.0, 10.0, 100.0)


def ranking(scores):
    """Collection order is numeric docid order: stable ties prefer lower docid."""
    return np.argsort(-np.asarray(scores), kind='stable')


def hybrid_scores(zone_scores, bm25_scores):
    """Ordinal scores: zone first, BM25 within ties, then numeric document ID.

    Round zone scores to 12 places to remove floating-point addition artifacts.
    Output scores encode rank only, not a weighted sum of zone and BM25 scores.
    """
    order = np.lexsort((np.arange(len(zone_scores)), -np.asarray(bm25_scores),
                        -np.round(zone_scores, 12)))
    scores = np.empty(len(order), dtype=float)
    scores[order] = np.arange(len(order), 0, -1)
    return scores


def metrics(scores, labels, k):
    hits = float(labels[ranking(scores)[:k]].sum())
    precision = hits / k
    recall = hits / labels.sum() if labels.sum() else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {'precision': precision, 'recall': recall, 'f1': f1}


def mean_f1(scores, labels, indices):
    return float(np.mean([metrics(scores[i], labels[i], 10)['f1'] for i in indices]))


def split_queries(n, seed):
    if n < 5:
        raise ValueError('Training requires at least five queries')
    permutation = np.random.default_rng(seed).permutation(n)
    a, b = int(.6 * n), int(.8 * n)
    return {name: np.sort(ids) for name, ids in
            zip(('train', 'validation', 'test'), np.split(permutation, [a, b]))}


def grid_weights(divisions=10):
    for a in range(divisions + 1):
        for b in range(divisions - a + 1):
            for c in range(divisions - a - b + 1):
                yield np.array([a, b, c, divisions-a-b-c]) / divisions


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def save_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_run(path, scores, queries, docnos, indices, top_k, key):
    with path.open('w') as handle:
        for i in indices:
            for rank, j in enumerate(ranking(scores[i])[:top_k], 1):
                handle.write(f'{queries[i][key]} Q0 {docnos[j]} {rank} '
                             f'{scores[i][j]:.9f} search_ninjas\n')


def train(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    documents, queries = read_documents(args.documents), read_queries(args.queries)
    qrels = read_qrels(args.qrels)
    collection = Collection(documents)
    known_docs = set(collection.docnos)
    if set(qrels) != {q['qid'] for q in queries}:
        raise ValueError('Training qrels must cover exactly the positional query ids')
    if any(not docs <= known_docs for docs in qrels.values()):
        raise ValueError('Qrels reference documents outside the collection')
    split = split_queries(len(queries), args.seed)
    save_json(args.output / 'search_ninjas_split.json', {
        name: [{'qid': queries[i]['qid'], 'original_id': queries[i]['original_id']} for i in ids]
        for name, ids in split.items()})
    labels = np.array([[doc in qrels[q['qid']] for doc in collection.docnos]
                       for q in queries], dtype=np.int8)
    print(f'Extracting 88 features for {len(queries)} queries x {len(documents)} documents', flush=True)
    extracted = [collection.features(q['text']) for q in queries]
    features = np.stack([x for x, _ in extracted])
    zones = np.stack([z for _, z in extracted])
    del extracted
    feature_seconds = time.perf_counter() - started
    best_score, best_weights, zone_trials = -1, None, []
    for weights in grid_weights():
        score = mean_f1(zones @ weights, labels, split['train'])
        zone_trials.append(dict(zip(FIELDS, weights.tolist()), train_f1_at_10=score))
        if score > best_score:
            best_score, best_weights = score, weights
    save_csv(args.output / 'search_ninjas_zone_grid.csv', zone_trials)
    print(f'Zone weights: {best_weights.tolist()}, train F1@10={best_score:.4f}', flush=True)
    train_x = features[split['train']].reshape(-1, features.shape[-1])
    train_y = labels[split['train']].ravel()
    if len(np.unique(train_y)) != 2:
        raise ValueError('SVM training needs both relevant and nonrelevant pairs')
    best_validation, best_model, best_alpha, svm_trials = -1, None, None, []
    for alpha in ALPHAS:
        model = make_pipeline(StandardScaler(), SGDClassifier(
            loss='hinge', penalty='l2', alpha=alpha, class_weight='balanced',
            average=True, max_iter=2000, tol=1e-4, n_iter_no_change=10,
            random_state=args.seed))
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            model.fit(train_x, train_y)
        validation_scores = model.decision_function(features.reshape(-1, 88)).reshape(labels.shape)
        score = mean_f1(validation_scores, labels, split['validation'])
        svm_trials.append({'alpha': alpha, 'validation_f1_at_10': score, 'epochs': int(model.steps[1][1].n_iter_)})
        print(f'SVM alpha={alpha}: validation F1@10={score:.4f}, epochs={model.steps[1][1].n_iter_}', flush=True)
        if score > best_validation:
            best_validation, best_model, best_alpha = score, model, alpha
    save_csv(args.output / 'search_ninjas_svm_grid.csv', svm_trials)
    scores = {
        'zone': zones @ best_weights,
        'svm': best_model.decision_function(features.reshape(-1, 88)).reshape(labels.shape),
        'equal_zone': zones.mean(2),
        'body_bm25': features[:, :, -1],
    }
    scores['zone_bm25_tiebreak'] = np.stack([
        hybrid_scores(z, b) for z, b in zip(scores['zone'], scores['body_bm25'])])
    zone_diagnostics = {
        'best_weight_tuples': sum(abs(r['train_f1_at_10'] - best_score) < 1e-12 for r in zone_trials),
        'body_match_fraction': float(zones[:, :, 3].mean()),
        'mean_distinct_scores': float(np.mean([len(np.unique(np.round(z, 12))) for z in scores['zone']])),
        'first_test_qid': queries[split['test'][0]]['qid'],
        'first_test_top_ties': int(np.sum(np.isclose(scores['zone'][split['test'][0]], scores['zone'][split['test'][0]].max(), rtol=0, atol=1e-12))),
    }
    save_json(args.output / 'search_ninjas_zone_diagnostics.json', zone_diagnostics)
    rows, aggregate = [], []
    for method, values in scores.items():
        for partition, indices in split.items():
            for k in CUTOFFS:
                measured = [metrics(values[i], labels[i], k) for i in indices]
                aggregate.append({'method': method, 'split': partition, 'k': k,
                                  **{m: float(np.mean([r[m] for r in measured]))
                                     for m in ('precision', 'recall', 'f1')}})
                for i, result in zip(indices, measured):
                    rows.append({'method': method, 'split': partition, 'k': k,
                                 'qid': queries[i]['qid'], **result})
        if method in ('zone', 'svm', 'zone_bm25_tiebreak'):
            for key, suffix in (('qid', 'seqid'), ('original_id', 'labels')):
                write_run(args.output / f'search_ninjas_{method}_test_{suffix}.txt', values,
                          queries, collection.docnos, split['test'], 100, key)
    save_csv(args.output / 'search_ninjas_metrics.csv', aggregate)
    save_csv(args.output / 'search_ninjas_per_query.csv', rows)
    scaler, svm = best_model.steps[0][1], best_model.steps[1][1]
    raw_weights = svm.coef_[0] / scaler.scale_
    save_csv(args.output / 'search_ninjas_svm_weights.csv', [
        {'feature': name, 'standardized_weight': float(weight), 'raw_weight': float(raw)}
        for name, weight, raw in zip(feature_names(), svm.coef_[0], raw_weights)])
    metadata = {
        'seed': args.seed, 'documents': len(documents), 'queries': len(queries),
        'split_sizes': {k: len(v) for k, v in split.items()},
        'features': feature_names(), 'zone_weights': dict(zip(FIELDS, best_weights.tolist())),
        'svm_alpha': best_alpha, 'svm_epochs': int(svm.n_iter_), 'svm_intercept': float(svm.intercept_[0]),
        'raw_intercept': float(svm.intercept_[0] - raw_weights @ scaler.mean_),
        'train_positive_pairs': int(train_y.sum()), 'train_pairs': len(train_y),
        'feature_seconds': feature_seconds, 'total_seconds': time.perf_counter() - started,
        'inputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (args.documents, args.queries, args.qrels)},
        'group': GROUP_DETAILS,
        'alpha_grid': list(ALPHAS),
        'role': 'evaluation',
        'fit_qids': [queries[i]['qid'] for i in split['train']],
        'revision_note': 'Expanded validation search and exploratory tie-break experiment after inspection of the original test results; this is not a fresh untouched test evaluation.',
    }
    save_json(args.output / 'search_ninjas_metadata.json', metadata)
    joblib.dump({'collection': collection, 'zone_weights': best_weights,
                 'svm': best_model, 'metadata': metadata}, args.output / 'search_ninjas_model.joblib')
    # A separate deployment bundle; neither its weights nor its predictions are
    # substituted into the evaluation metrics above.
    final_ids = np.sort(np.concatenate((split['train'], split['validation'])))
    final_model = clone(best_model)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        final_model.fit(features[final_ids].reshape(-1, 88), labels[final_ids].ravel())
    final_best, final_weights = -1, None
    for weights in grid_weights():
        value = mean_f1(zones @ weights, labels, final_ids)
        if value > final_best:
            final_best, final_weights = value, weights
    final_metadata = {
        'role': 'deployment', 'group': GROUP_DETAILS, 'seed': args.seed,
        'fit_qids': [queries[i]['qid'] for i in final_ids],
        'excluded_test_qids': [queries[i]['qid'] for i in split['test']],
        'train_pairs': int(len(final_ids) * len(documents)),
        'svm_alpha': best_alpha,
        'svm_epochs': int(final_model.steps[1][1].n_iter_),
        'zone_weights': dict(zip(FIELDS, final_weights.tolist())),
        'inputs_sha256': metadata['inputs_sha256'],
        'evaluation': 'No reported test metrics belong to this refitted model.',
    }
    save_json(args.output / 'search_ninjas_final_metadata.json', final_metadata)
    joblib.dump({'collection': collection, 'zone_weights': final_weights,
                 'svm': final_model, 'metadata': final_metadata},
                args.output / 'search_ninjas_final_model.joblib')
    for row in aggregate:
        if row['split'] == 'test' and row['k'] == 10:
            print(row)
    from search_ninjas_report import generate_report
    generate_report(args.output)
    print(f'Artifacts and report saved to {args.output}', flush=True)


def predict(args):
    # joblib uses pickle: load only a model produced locally or from a trusted source.
    bundle = joblib.load(args.model)
    queries = read_queries(args.queries)
    collection = bundle['collection']
    args.output.mkdir(parents=True, exist_ok=True)
    for method in ('zone', 'svm'):
        scores = []
        for q in queries:
            features, zones = collection.features(q['text'])
            scores.append(zones @ bundle['zone_weights'] if method == 'zone'
                          else bundle['svm'].decision_function(features))
        for key, suffix in (('qid', 'seqid'), ('original_id', 'labels')):
            write_run(args.output / f'search_ninjas_{method}_{suffix}.txt', scores,
                      queries, collection.docnos, range(len(queries)), args.top_k, key)
    print(f'Ranked {len(queries)} queries with both methods; output: {args.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    training = sub.add_parser('train', help='run the full reproducible experiment')
    training.add_argument('--documents', type=Path, default=BASE / 'cran.all.1400')
    training.add_argument('--queries', type=Path, default=BASE / 'cran.qry')
    training.add_argument('--qrels', type=Path, default=BASE / 'cranqrel')
    training.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    training.add_argument('--seed', type=int, default=42)
    training.set_defaults(function=train)
    prediction = sub.add_parser('predict', help='rank unseen queries without relevance judgments')
    prediction.add_argument('--model', type=Path, default=DEFAULT_OUTPUT / 'search_ninjas_final_model.joblib')
    prediction.add_argument('--queries', type=Path, required=True)
    prediction.add_argument('--output', type=Path, default=DEFAULT_OUTPUT / 'predictions')
    prediction.add_argument('--top-k', type=int, default=100)
    prediction.set_defaults(function=predict)
    args = parser.parse_args()
    if getattr(args, 'top_k', 1) < 1:
        parser.error('--top-k must be positive')
    with threadpool_limits(limits=2):
        args.function(args)


if __name__ == '__main__':
    main()
