"""Small regressions for failure-prone native-label conversions."""
import unittest
from adapters import mathqa_options, mcq, native_split
from common import row, validate
from prepare import NearIndex

class ConversionTests(unittest.TestCase):
    def test_mathqa_number_commas_do_not_split_choices(self):
        options = mathqa_options('a ) 1,200 , b ) 3,400 , c ) 5 , d ) 6 , e ) none of these')
        self.assertEqual(options, ['1,200', '3,400', '5', '6', 'none of these'])

    def test_arc_numeric_labels_use_native_key(self):
        source = {'name':'arc-challenge', 'category':'science', 'adapter':'arc', 'repo':'test', 'revision':'test'}
        raw = {'id':'q', 'question':'Q?', 'choices':{'label':['1','2','3','4'],
                'text':['first','second','third','fourth']}, 'answerKey':'3'}
        converted = mcq(source, raw, 0, 'train', 'train.parquet')
        self.assertEqual(converted['target'], [0.,0.,1.,0.])
        self.assertEqual(converted['options'], raw['choices']['text'])
        validate(converted)

    def test_benchmark_validation_is_not_development(self):
        self.assertEqual(native_split('winogrande', 'validation-0.parquet'), 'test')
        self.assertEqual(native_split('arc-challenge', 'validation-0.parquet'), 'validation')
        self.assertEqual(native_split('mmlu', 'auxiliary_train-0.parquet'), 'train')
        self.assertEqual(native_split('mmlu', 'dev-0.parquet'), 'test')

    def test_duplicate_candidates_are_rejected(self):
        source = {'name':'s', 'category':'math', 'repo':'test', 'revision':'test'}
        converted = row(source, 'a', 'train', 'Q?', ['Same', ' same '], 0, 'math.test')
        with self.assertRaisesRegex(ValueError, 'duplicate_options'):
            validate(converted)

    def test_near_audit_detects_small_extension(self):
        index = NearIndex()
        index.add('A very long original question asking which chemical reaction produces oxygen during photosynthesis in green plants', 'protected')
        self.assertEqual(index.match('A very long original question asking which chemical reaction produces oxygen during photosynthesis in green plants today'), 'protected')
        self.assertIsNone(index.match('An unrelated numerical question on compound interest and banking'))

    def test_gsm_native_train_and_test_ids_do_not_collide(self):
        source = {'name':'gsm8k','adapter':'gsm','category':'math','repo':'test','revision':'test'}
        raw = {'question':'How many?', 'answer':'reasoning\n#### 7'}
        train = mcq(source, raw, 0, 'train', 'train.parquet')
        test = mcq(source, raw, 0, 'test', 'test.parquet')
        self.assertNotEqual(train['id'], test['id'])
        self.assertEqual(train['metadata']['native_id'], '0')
        self.assertNotIn('reasoning', train['state'])

if __name__ == '__main__':
    unittest.main()
