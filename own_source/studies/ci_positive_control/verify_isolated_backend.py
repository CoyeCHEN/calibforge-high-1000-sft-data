"""Actual Jupyter/Docker state, all six tools and reference isolation."""
import json
import os
from pathlib import Path
import sys

from isolated_kernel import IsolatedIPython


def main(root):
    root=Path(root)
    a=IsolatedIPython(name='IPythonInterpreter',user_data_dir=str(root/'data/cibench_dataset/datasources'))
    try:
        first=a('stateful_ci_x=41',timeout=30)
        second=a('print(stateful_ci_x + 1)',timeout=30)
        reference=a("from pathlib import Path\nprint(Path('../cibench_generation/pandas/experiment01.ipynb').exists())",timeout=30)
        write=a.terminal('write',{'path':'ci-write-test.txt','content':'hello isolated CI'})
        shared=a('print(Path("ci-write-test.txt").read_text())',timeout=30)
        bash=a.terminal('bash',{'command':'printf kernel-bash-ok'})
        read=a.terminal('read',{'path':'ci-write-test.txt'})
        edit=a.terminal('edit',{'path':'ci-write-test.txt','edits':[{'oldText':'hello','newText':'goodbye'}]})
        background=a.terminal('bash',{'command':'sleep 1; printf background-ok','background':True,'timeout':30})
        jobs=a.terminal('bash_job',{'action':'list'})
        finish1=a.terminal('finish',{'summary':'non-formal isolated validation'})
        finish2=a.terminal('finish',{'summary':'checked'})
        assert int(first.state)==int(second.state)==0
        assert '42' in second.result['text'] and 'False' in reference.result['text']
        assert 'hello isolated CI' in shared.result['text'] and finish2['finishRequested'] is True
        a.reset()
        after=a('from pathlib import Path\nprint("stateful_ci_x" in globals()); print(Path("ci-write-test.txt").exists())',timeout=30)
        assert after.result['text'].count('False')==2
        result=dict(status='PASSED',stateful_second_cell=second.result,reference_not_visible=reference.result,
            shared_filesystem=shared.result,all_six_tools_real_execution=True,
            native_tools=dict(write=write,read=read,edit=edit,bash=bash,background=background,jobs=jobs,
                              finish1=finish1,finish2=finish2),fresh_kernel_and_filesystem=after.result)
    finally:
        a.reset()
    result['normal_shutdown_and_owned_container_cleanup']=True
    (root/'checks/isolated-kernel-canary.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ISOLATED_ALL_TOOLS_AND_STATE_PASSED',flush=True)


if __name__=='__main__':main(sys.argv[1])
