import unittest

from groundbeef_v05_build import parse_openscience, QUOTAS, validate_example
from v05_code import execute_menu, normalized_program


class V05Tests(unittest.TestCase):
    def test_native_menu_and_solution_boundary(self):
        raw={'input':'Which number is even?\nA: 3\nB: 4\nC: 5\nD: 7',
             'output':'<think>Private reasoning with answer A mentioned.</think>\\boxed{B}'}
        r=parse_openscience(raw,'test',0)
        self.assertEqual(r['target'],[0.,1.,0.,0.])
        self.assertEqual(r['options'],['3','4','5','7'])
        self.assertNotIn('Private',r['state']);self.assertEqual(sum(QUOTAS.values()),32000)
        raw['output']='<think>x</think>The answer is A. \\boxed{B}'
        with self.assertRaises(ValueError):parse_openscience(raw,'test',0)
        raw['output']='<think>x</think>The answer is (ABCD). Forced conclusion: \\boxed{A}'
        with self.assertRaises(ValueError):parse_openscience(raw,'test',0)
        raw['input']+='\nA: second menu'
        with self.assertRaises(ValueError):parse_openscience(raw,'test',0)

    def test_normalized_wrapper_parent(self):
        a='def f(x):\n    return x[::-1]'
        b='def main_solution(x):\n    return x[::-1]\n\nf = main_solution'
        self.assertEqual(normalized_program(a),normalized_program(b))

    def test_python_candidate_case_and_spaces_are_meaningful(self):
        r={'domain':'code','metadata':{'dataset_source':'codeio-short'},'state':{'code':'x'},'question':'output?','options':["'A'","'a'","'a  b'","'a b'"],
           'target':[1.,0.,0.,0.]}
        validate_example(r)
        r['options'][1]='"A"'
        with self.assertRaises(ValueError):validate_example(r)

    def test_executed_semantic_negatives_and_gold_witness(self):
        task={'id':'test','code':"def f(a):\n    return ' '.join(a.split())",'function':'f',
              'input':[' h e l l o  w o r l d '],'expected_json':'"h e l l o w o r l d"'}
        result=execute_menu(task)
        self.assertNotIn('rejection',result)
        self.assertEqual(result['options'][result['gold']],"'h e l l o w o r l d'")
        self.assertTrue(any(o['origin'].startswith('mutant:join_separator') for o in result['origins']))
        task['expected_json']='"wrong"'
        self.assertIn('rejection',execute_menu(task))

    def test_inplace_mutation_keeps_original_input_and_matches_request_tag(self):
        task={'id':'mutating-list','code':'def f(xs):\n    return xs.pop(0)',
              'function':'f','input':[[7,4,1]],'expected_json':'7',
              'other_inputs':[{'input':[[2,9]],'output':2}]}
        result=execute_menu(task)
        self.assertNotIn('rejection',result)
        self.assertEqual(result['execution']['invocation'],'f([7, 4, 1])')
        self.assertEqual(result['options'][result['gold']],'7')
        self.assertIn('2',result['options'])
        self.assertEqual(task['input'],[[7,4,1]])

    def test_tuple_input_survives_worker_transport(self):
        task={'id':'tuple-input','code':'def f(xs):\n    return (isinstance(xs, tuple), len(xs))',
              'function':'f','input':[(1,2)],'expected_json':'[true,2]',
              'other_inputs':[{'input':[(1,)],'output':(True,1)}]}
        result=execute_menu(task)
        self.assertNotIn('rejection',result)
        self.assertEqual(result['execution']['invocation'],'f((1, 2))')
        self.assertEqual(result['options'][result['gold']], '(True, 2)')


if __name__=='__main__':unittest.main()
