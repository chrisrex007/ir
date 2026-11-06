"""22 retrieval features per document field, without relevance information."""
from collections import Counter
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from search_ninjas_data import FIELDS, tokens

FEATURES = ('query_term_number', 'query_term_ratio', 'stream_length', 'idf',
            'tf_sum', 'tf_min', 'tf_max', 'tf_mean', 'tf_variance',
            'normalized_tf_sum', 'normalized_tf_min', 'normalized_tf_max',
            'normalized_tf_mean', 'normalized_tf_variance',
            'tfidf_sum', 'tfidf_min', 'tfidf_max', 'tfidf_mean', 'tfidf_variance',
            'boolean', 'vector_space', 'bm25')


def moments(values):
    if values.shape[1] == 0:
        return np.zeros((values.shape[0], 5))
    return np.column_stack((values.sum(1), values.min(1), values.max(1),
                            values.mean(1), values.var(1)))


class Collection:
    def __init__(self, documents):
        self.docnos = [d['docno'] for d in documents]
        self.fields = []
        for field in FIELDS:
            counts = [Counter(tokens(d[field])) for d in documents]
            vocab = {t: i for i, t in enumerate(sorted(set().union(*counts)))}
            rows, cols, data = [], [], []
            for row, counter in enumerate(counts):
                for term, frequency in counter.items():
                    rows.append(row)
                    cols.append(vocab[term])
                    data.append(frequency)
            tf = csr_matrix((np.asarray(data, dtype=float), (rows, cols)),
                            shape=(len(documents), len(vocab)))
            lengths = np.asarray(tf.sum(1)).ravel()
            df = np.asarray((tf > 0).sum(0)).ravel()
            idf = np.log((len(documents) + 1) / (df + 1)) + 1
            bm25_idf = np.log1p((len(documents) - df + .5) / (df + .5))
            weighted = tf.copy()
            weighted.data = 1 + np.log(weighted.data)
            weighted = (normalize(weighted.multiply(idf), norm='l2').tocsr()
                        if vocab else weighted)
            self.fields.append((vocab, tf, lengths, idf, bm25_idf, weighted))

    def features(self, text):
        query = Counter(tokens(text))
        terms = sorted(query)
        blocks, zones = [], []
        n = len(self.docnos)
        for vocab, tf, lengths, idf, bm_idf, vectors in self.fields:
            # Include zero columns for OOV terms in all statistics and ratios.
            frequencies = np.zeros((n, len(terms)))
            query_idf = np.full(len(terms), np.log(n + 1) + 1)
            bm = np.zeros(len(terms))
            qvec = np.zeros(len(vocab))
            for j, term in enumerate(terms):
                if term in vocab:
                    col = vocab[term]
                    frequencies[:, j] = tf[:, col].toarray().ravel()
                    query_idf[j], bm[j] = idf[col], bm_idf[col]
                    qvec[col] = (1 + np.log(query[term])) * idf[col]
            matched = (frequencies > 0).sum(1)
            boolean = (matched > 0).astype(float)
            ratio = matched / max(len(terms), 1)
            zones.append(ratio)
            norm_tf = frequencies / np.maximum(lengths[:, None], 1)
            qnorm = np.linalg.norm(qvec)
            cosine = vectors @ (qvec / qnorm) if qnorm else np.zeros(n)
            avg_length = max(lengths.mean(), 1e-12)
            denominator = frequencies + 1.2 * (.25 + .75 * lengths[:, None] / avg_length)
            bm25 = (bm * frequencies * 2.2 / denominator).sum(1)
            blocks.append(np.column_stack((matched, ratio,
                lengths, (frequencies > 0) @ query_idf, moments(frequencies),
                moments(norm_tf), moments(frequencies * query_idf), boolean, cosine, bm25)))
        return np.hstack(blocks).astype(np.float32), np.column_stack(zones).astype(np.float32)


def feature_names():
    return [f'{field}.{feature}' for field in FIELDS for feature in FEATURES]
