"""Regression checks for transformations that can silently corrupt supervision."""
import copy
import unittest

from groundbeef import aliases, decorate, eval_format, parse_call, prompt, quick_quotas, visible_hash


class GroundbeefTests(unittest.TestCase):
    def test_nli_keeps_label_mapping_when_moving_text(self):
        r = {'id':'x','group_id':'p','source':'anli','kind':'choice','state':'{"premise":"A", "hypothesis":"B"}',
             'question':'relation','options':['Entailment','Neutral','Contradiction'],'target':[0.,1.,0.],
             'domain':'grounding','family':'nli','metadata':{'native_round':'3'}}
        out = decorate(r,'anli','validation')
        self.assertEqual(out['target'],[0.,1.,0.])
        self.assertEqual(out['options'],['entailment','neutral','contradiction'])
        self.assertIn('Premise: A\nHypothesis: B',out['question'])
        self.assertEqual(out['state'],{})

    def test_toolace_names_with_spaces_and_slashes(self):
        tools = [{'name':'Get Languages for Country','parameters':{'properties':{'country':{}},'required':['country']}},
                 {'name':'/madlibs-diceware','parameters':{'properties':{'nphrase':{}},'required':['nphrase']}}]
        self.assertEqual(parse_call('[Get Languages for Country(country="BR")]',tools),(0,{'country':'BR'}))
        self.assertEqual(parse_call('[/madlibs-diceware(nphrase=2)]',tools),(1,{'nphrase':2}))
        self.assertIsNone(parse_call('Could you supply the country?',tools))

    def test_toolace_ambiguous_or_nonliteral_calls_rejected(self):
        tools = [{'name':'foo','parameters':{'properties':{'x':{}},'required':['x']}}]
        for text in ['[foo()]','[foo(x=1), foo(x=2)]','[foo(x=__import__("os"))]','[foo(z=2)]']:
            with self.assertRaises((ValueError,SyntaxError)):
                parse_call(text,tools)

    def test_metadata_and_gold_never_enter_prompt(self):
        r = {'state':{'question':'actual request'},'question':'choose','options':['a','b'],
             'target':[1.,0.],'metadata':{'secret_gold':'NEVER_VISIBLE'}}
        other = copy.deepcopy(r);other['target']=[0.,1.]
        self.assertEqual(prompt(r),prompt(other))
        self.assertNotIn('NEVER_VISIBLE',prompt(r))

    def test_aliases_ignore_serialisation_and_candidates(self):
        r = {'state':{'request':'Pay the invoice'},'question':'choose','options':['a','b']}
        other = {'state':'{"request": "Pay the invoice"}','question':'choose','options':['b','a']}
        self.assertEqual(aliases(r),aliases(other))
        self.assertEqual(visible_hash(r),visible_hash(other))

    def test_rag_remaps_supported_gold_to_benchmark_false(self):
        r = {'state':{'prompt':'facts','response':'answer','task':'QA'},'options':['Supported','Unsupported'],
             'question':'check','target':[1.,0.],'metadata':{}}
        eval_format(r,'ragtruth')
        self.assertEqual(r['target'],[0.,1.])
        self.assertIn('All content',r['options'][1])
        self.assertEqual(r['metadata']['source_candidate_mapping'],[1,0])
        self.assertEqual(r['state'],{'prompt':'facts','response':'answer'})

    def test_quick_set_keeps_source_proportions(self):
        recipe = {'validation':{'many_tasks':300,'one_task':3700},'quick_validation_rows':1000}
        self.assertEqual(quick_quotas(recipe),{'many_tasks':75,'one_task':925})


if __name__ == '__main__':
    unittest.main()
