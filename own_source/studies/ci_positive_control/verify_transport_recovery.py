"""Prove the parser repair leaves every observed legal M0 decision unchanged."""
import ast
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import textwrap


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);value=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value);return value


def main(root):
    root=Path(root)
    old=module('old_ci_transport',root/'source/evaluation-frozen-r5/cibench_adapter.py')
    new=module('new_ci_transport',root/'source/sft/studies/ci_positive_control/cibench_adapter.py')
    names=[t['function']['name']for t in json.loads((root/'configs/tools.json').read_text())]
    executor=SimpleNamespace(is_valid=lambda name:name in names,
                             invalid_action=SimpleNamespace(name='InvalidAction'))
    assert ast.dump(ast.parse(textwrap.dedent(inspect.getsource(old.NativeProtocol.format))))==ast.dump(
        ast.parse(textwrap.dedent(inspect.getsource(new.NativeProtocol.format))))
    count=0
    for line in (root/'logs/requests-m0-isolated-r2.jsonl').open():
        message=json.loads(line)['response']['choices'][0]['message']
        pairs=[]
        for implementation in [old,new]:
            packet=implementation.NativeAssistant(json.dumps(message,ensure_ascii=False));packet.message=message
            pairs.append(implementation.NativeProtocol().parse(packet,executor))
        assert pairs[0]==pairs[1],(count,pairs)
        count+=1
    assert count>0
    (root/'checks/transport-recovery-prefix-equivalence.json').write_text(json.dumps(dict(status='PASSED',
        real_recorded_model_decisions=count,all_legal_parse_results_equal=True,format_ast_unchanged=True,
        scope='Unknown tool names become explicit error observations; no alias, correction prompt, or new retry'),indent=2)+'\n')
    print('OBSERVED_LEGAL_DECISIONS_UNCHANGED',count,flush=True)


if __name__=='__main__':main(sys.argv[1])
