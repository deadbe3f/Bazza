"""Restricted, disposable Python I/O verification worker; launch inside Docker.

AST restrictions are a first filter, not an OS security boundary. The parent
launcher also disables network, drops capabilities, and mounts original data
read-only. Resource limits and a parent wall-clock timeout bound each program.
"""
import ast
import builtins
import json
import resource
import sys

MODULES = {"math", "collections", "itertools", "functools", "operator", "re", "string", "heapq", "bisect"}
BUILTINS = {"abs", "all", "any", "bin", "bool", "chr", "complex", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", "hex", "int", "isinstance", "iter", "len", "list", "map", "max", "min", "next", "oct", "ord", "pow", "range", "repr", "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError", "TypeError", "IndexError", "KeyError", "ZeroDivisionError", "StopIteration"}


def check(code):
    if len(code) > 2400:
        raise ValueError("code too long")
    tree = ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n, (ast.ClassDef, ast.AsyncFunctionDef, ast.Await, ast.Global, ast.Nonlocal)):
            raise ValueError("unsupported syntax")
        if isinstance(n, ast.Name) and n.id.startswith("__"):
            raise ValueError("dunder name")
        if isinstance(n, ast.Attribute) and n.attr.startswith("_"):
            raise ValueError("private attribute")
        if isinstance(n, ast.Import) and any(x.name not in MODULES for x in n.names):
            raise ValueError("unapproved import")
        if isinstance(n, ast.ImportFrom) and (n.level or n.module not in MODULES or any(x.name.startswith("_") or x.name == "*" for x in n.names)):
            raise ValueError("unapproved import")
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in {
            "eval", "exec", "open", "input", "compile", "getattr", "setattr", "delattr", "globals", "locals", "vars", "dir", "type", "object", "breakpoint", "help"}:
            raise ValueError("dynamic capability")
    return tree


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in MODULES or level or any(s.startswith("_") for s in (fromlist or ())):
        raise ImportError("unapproved import")
    return builtins.__import__(name, globals, locals, fromlist, level)


def call(code, function, supplied):
    tree = check(code)
    namespace = {"__builtins__": {**{k: getattr(builtins, k) for k in BUILTINS}, "__import__": safe_import}}
    exec(compile(tree, "<qualified-source>", "exec"), namespace)
    fn = namespace[function]
    if not callable(fn):
        raise ValueError("entrypoint is not callable")
    if isinstance(supplied, dict):
        value = fn(**supplied)
        invocation = "f(" + ", ".join(k + "=" + repr(v) for k, v in supplied.items()) + ")"
    elif isinstance(supplied, list):
        value = fn(*supplied)
        invocation = "f(" + ", ".join(repr(v) for v in supplied) + ")"
    else:
        value = fn(supplied)
        invocation = "f(" + repr(supplied) + ")"
    literal = repr(value)
    parsed = ast.literal_eval(literal)
    if parsed != value or len(literal) > 2000:
        raise ValueError("nonliteral or large result")
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return {"literal": literal, "json": serialized, "invocation": invocation}


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_AS, (256 * 2**20, 256 * 2**20))
    resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    data = json.load(sys.stdin)
    try:
        result = call(data["code"], data["function"], data["input"])
        print(json.dumps({"ok": True, **result}, ensure_ascii=False, allow_nan=False))
    except BaseException as e:
        print(json.dumps({"ok": False, "error": type(e).__name__ + ": " + str(e)[:150]}))


if __name__ == "__main__":
    main()
