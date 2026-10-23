"""Disposable JSON worker. Run only inside the network-disabled CPU container."""
import ast
import json
import resource
import sys

from python_execution_worker import call
from v05_code import invocation


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    for item in json.load(sys.stdin):
        try:
            supplied=ast.literal_eval(item['input_literal'])
            before=repr(supplied)
            rendered=invocation(supplied)
            result = call(item['code'], item['function'], supplied)
            # The legacy helper renders after the call; replace that rendering.
            result['invocation']=rendered
            out = {'ok': True, 'tag': item['tag'], 'supplied_input_literal': before, **result}
        except BaseException as e:
            out = {'ok': False, 'tag': item['tag'], 'error': type(e).__name__ + ': ' + str(e)[:120]}
        print(json.dumps(out, ensure_ascii=False, allow_nan=False), flush=True)


if __name__ == '__main__': main()
