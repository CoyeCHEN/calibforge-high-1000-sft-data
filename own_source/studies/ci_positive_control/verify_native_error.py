"""Run an actual native Bash failure and recover in the same Jupyter kernel."""
import argparse
import json
import os
from pathlib import Path

from cibench_adapter import HarnessAction, NativeProtocol
from isolated_kernel import IsolatedIPython
from tool_contract import terminal_tools


def main(root):
    root = Path(root)
    python = IsolatedIPython(name='IPythonInterpreter',
                             user_data_dir=str(root / 'data/cibench_dataset/datasources'))
    try:
        initialized = python('ci_error_recovery_value = 41', timeout=30)
        assert int(initialized.state) == 0
        tools, _ = terminal_tools(root, Path(os.environ['CI_EXECUTION_WORKDIR']))
        bash = HarnessAction(next(tool for tool in tools if tool['name'] == 'bash'), python)
        failed = bash(command='printf expected-error; exit 7')
        assert int(failed.state) != 0 and 'expected-error' in failed.errmsg
        protocol = NativeProtocol()
        protocol.last_call = dict(id='native-error-canary')
        observation = json.loads(protocol.format_response(failed))
        assert observation['role'] == 'tool' and observation['content'] == failed.errmsg
        recovered = python('print(ci_error_recovery_value + 1)', timeout=30)
        assert int(recovered.state) == 0 and '42' in recovered.result['text']
        result = dict(status='PASSED', failed_action_state=int(failed.state),
                      error_observation=observation, recovered_python=recovered.result,
                      same_stateful_kernel=True, benchmark_task=False)
    finally:
        python.reset()
    with (root / 'checks/native-error-recovery-fixed.json').open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    main(parser.parse_args().root)
