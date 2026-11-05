import tempfile
from pathlib import Path
import unittest
import numpy as np
from search_ninjas_data import FIELDS, read_documents, read_queries, read_qrels
from search_ninjas_features import Collection, feature_names
from search_ninjas_rank import grid_weights, metrics, ranking, split_queries, write_run


class RankingTests(unittest.TestCase):
    def fixture(self, text):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / 'data'
        path.write_text(text)
        return path

    def test_parsing_tags_empty_documents_and_query_ids(self):
        docs = read_documents(self.fixture('.I 1\n.T\nwing\n.A\nauthor\n.W\nflow\n.A\ncontinued\n.W\nagain\n.I 2\n.T\n.A\n.B\n.W\n'))
        self.assertIn('continued', docs[0]['body'])
        self.assertIn('again', docs[0]['body'])
        self.assertEqual(docs[0]['author'].strip(), 'author')
        self.assertEqual(docs[1]['body'], '')
        queries = read_queries(self.fixture('.I 001\n.W\nwing\n.I 008\n.W\nflow\n'))
        self.assertEqual(queries[1]['qid'], '2')
        self.assertEqual(queries[1]['original_id'], '008')
        qrels = read_qrels(self.fixture('1 1 4\n1 2 -1\n2 2 1\n2 1 5\n'))
        self.assertEqual(qrels, {'1': {'1'}, '2': {'2'}})

    def test_feature_values_and_empty_queries(self):
        docs = [{'docno': str(i+1), **{f: '' for f in FIELDS}} for i in range(2)]
        docs[0]['title'] = 'wing wing flow'
        docs[1]['title'] = 'flow'
        collection = Collection(docs)
        x, zones = collection.features('wing missing')
        self.assertEqual(x.shape, (2, 88))
        self.assertEqual(len(feature_names()), 88)
        np.testing.assert_allclose(x[0, :3], [1, .5, 3])
        np.testing.assert_allclose(x[0, 4:9], [2, 0, 2, 1, 1])
        np.testing.assert_allclose(x[0, 9:14], [2/3, 0, 2/3, 1/3, 1/9])
        idf = 1 + np.log(3/2)
        self.assertAlmostEqual(float(x[0, 14]), 2*idf, places=5)
        expected_bm = np.log(2) * 2 * 2.2 / (2+1.2*(.25+.75*3/2))
        self.assertAlmostEqual(float(x[0, 21]), expected_bm, places=5)
        self.assertEqual(zones.tolist(), [[1,0,0,0], [0,0,0,0]])
        empty, z = collection.features('the and')
        self.assertTrue(np.isfinite(empty).all())
        self.assertEqual(z.sum(), 0)
        self.assertEqual(empty[:, 19:22].sum(), 0)
        cosine_features, _ = collection.features('wing flow')
        doc_wing = (1 + np.log(2)) * idf
        expected_cosine = (doc_wing * idf + 1) / (np.hypot(doc_wing, 1) * np.hypot(idf, 1))
        self.assertAlmostEqual(float(cosine_features[0, 20]), expected_cosine, places=6)

    def test_supplied_collection(self):
        root = Path(__file__).resolve().parents[1]
        documents = read_documents(root / 'cran.all.1400')
        queries = read_queries(root / 'cran.qry')
        judgments = read_qrels(root / 'cranqrel')
        self.assertEqual((len(documents), len(queries), len(judgments)), (1400, 225, 225))
        self.assertEqual(queries[2]['original_id'], '004')
        self.assertEqual(queries[2]['qid'], '3')
        self.assertEqual([d['docno'] for d in documents if not any(d[f].strip() for f in FIELDS)],
                         ['471', '995'])

    def test_metrics_and_ties(self):
        scores = np.array([2, 2, 1, 0])
        np.testing.assert_array_equal(ranking(scores), [0,1,2,3])
        result = metrics(scores, np.array([1,0,1,0]), 2)
        self.assertEqual(result, {'precision': .5, 'recall': .5, 'f1': .5})
        self.assertEqual(metrics(scores, np.zeros(4), 2)['f1'], 0)

    def test_grid_and_no_query_leakage(self):
        weights = np.array(list(grid_weights()))
        self.assertEqual(weights.shape, (286, 4))
        self.assertTrue((weights >= 0).all())
        np.testing.assert_allclose(weights.sum(1), 1)
        split = split_queries(225, 42)
        self.assertEqual([len(x) for x in split.values()], [135,45,45])
        self.assertEqual(len(set(np.concatenate(list(split.values())))), 225)
        for key, values in split.items():
            np.testing.assert_array_equal(values, split_queries(225, 42)[key])

    def test_run_uses_requested_id_convention(self):
        path = self.fixture('')
        queries = [{'qid':'2', 'original_id':'008'}]
        write_run(path, [[.1, .9]], queries, ['3','7'], [0], 1, 'original_id')
        self.assertEqual(path.read_text().split()[:4], ['008', 'Q0', '7', '1'])


if __name__ == '__main__':
    unittest.main()
