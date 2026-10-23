import json
from pathlib import Path
import tempfile
import unittest

from common import key
from decider_supplement import Groups, Protection, input_texts, record, sketch, teacher_agrees
from groundbeef import visible_hash


class NoveltyTests(unittest.TestCase):
    def test_teacher_adjacent_score_and_routing_exceptions_are_rejected(self):
        self.assertFalse(teacher_agrees({'type':'score','answer':2,'teacher_ok':True,'teacher_pred':3},2))
        self.assertFalse(teacher_agrees({'type':'choice','answer':'support','teacher_ok':False,'teacher_pred':'other'},0))
        self.assertTrue(teacher_agrees({'type':'score','answer':2,'teacher_p':.8},2))
        self.assertFalse(teacher_agrees({'type':'score','answer':2,'teacher_p':.5001},2))
        self.assertFalse(teacher_agrees({'type':'score','answer':2},2))

    def sample(self):
        return record({'name':'new','repo':'example','revision':'pinned'},'1','parent',
                      'A customer reported a duplicate charge on their card yesterday.',
                      'Which team?', ['billing','technical'],0,'routing')

    def test_option_order_and_target_do_not_hide_existing_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Protection(Path(tmp)/'p.sqlite');r=self.sample();p.add(r,'old');p.finish()
            r['options'].reverse();r['target'].reverse()
            self.assertEqual(p.match(r)[0],'existing_complete_input')
            r['question']='A different question about the same old message'
            self.assertEqual(p.match(r)[0],'existing_or_protected_context')
            p.db.close()

    def test_structured_request_alias_protects_reformatting(self):
        r=self.sample();raw=r['state'];r['state']=json.dumps({'request':raw,'schema':'new'})
        self.assertIn(key(raw),{key(t) for t in input_texts(r)})

    def test_sentence_pair_aliases_and_group_transitivity(self):
        r=self.sample();r['state']=json.dumps({'text1':'Sentence number one.','text2':'Sentence number two.'})
        self.assertIn('sentence number one.',input_texts(r))
        groups=Groups();groups.union('a','b');groups.union('b','c')
        self.assertEqual(groups.find('a'),groups.find('c'))

    def test_near_match_is_checked_after_sketch_lookup(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Protection(Path(tmp)/'p.sqlite');r=self.sample()
            r['state']=' '.join('term'+str(i) for i in range(80));p.add(r,'old');p.finish()
            r['state']+=' additional';r['question']='Changed question'
            self.assertEqual(p.match(r)[0],'near_existing_or_protected_context')
            r['state']='An entirely unrelated customer request with many completely different words.'
            self.assertIsNone(p.match(r));p.db.close()


if __name__=='__main__': unittest.main()
