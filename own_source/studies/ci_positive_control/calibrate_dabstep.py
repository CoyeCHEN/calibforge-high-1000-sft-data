"""Public answered dev calibration with upstream ReAct and a real shared kernel.

This is a transport/execution calibration, not the published DABStep leaderboard.
The answer key is retained only by this controller, never sent to the agent.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from lagent.actions import ActionExecutor
from lagent.agents.react import ReAct
from cibench_adapter import NativeTransport,NativeProtocol,HarnessAction,FinalAnswerAction
from isolated_kernel import IsolatedIPython
from tool_contract import terminal_tools


class CalibrationProtocol(NativeProtocol):
    def format(self,chat_history,inner_step,action_executor,force_stop=False):
        # Keep the original bounded loop. No last-turn correction prompt.
        return super().format(chat_history,inner_step,action_executor,force_stop=False)


def copy_public_context(root,interpreter):
    interpreter.initialize()
    name='qwen35-ci-kernel-'+Path(interpreter.kernel_manager.connection_file).stem
    def docker(args):
        return subprocess.run(['sg','docker','-c',shlex.join(['docker']+args)],
            capture_output=True,text=True,check=True,timeout=30).stdout
    labels=json.loads(docker(['inspect',name,'--format','{{json .Config.Labels}}']))
    assert labels['chenkeyi.run']=='qwen35-ci-positive-control-20261108'
    assert labels['chenkeyi.phase']==os.environ['CI_PHASE']
    # Only the seven released public assets enter the sandbox. No dev answers.
    docker(['exec','--user','0',name,'mkdir','-p','/home/user/input'])
    folder=root/'data/dabstep/context-release/data/context'
    assert len(list(folder.iterdir()))==7
    for file in folder.iterdir():
        docker(['cp',str(file),name+':/home/user/input/'+file.name])


def main(root,model,run_id,endpoint):
    root=Path(root);os.environ['CI_ROOT']=str(root)
    rows=[json.loads(x)for x in (root/'data/dabstep/dev.jsonl').open()]
    output=root/f'checks/dabstep-calibration-{run_id}.jsonl'
    if output.exists():raise FileExistsError('Calibration evidence is immutable')
    tools,_=terminal_tools(root,root/'runtime/dabstep-controller-unused')
    llm=NativeTransport(str(root),endpoint,'qwen35-ci-'+model)
    with output.open('x') as handle:
        for task in rows:
            python=IsolatedIPython(name='IPythonInterpreter',user_data_dir=str(root/'data/cibench_dataset/datasources'))
            try:
                copy_public_context(root,python)
                actions=[python]+[HarnessAction(tool,python)for tool in tools]
                executor=ActionExecutor(actions,finish_action=FinalAnswerAction(),finish_in_action=True)
                agent=ReAct(llm,executor,protocol=CalibrationProtocol(),max_turn=12)
                prompt=('The public task data are in /home/user/input. Read the relevant documentation and use '
                    'the stateful Python tool to compute your answer. Submit it with final_answer.\n\n'
                    +task['question']+'\n\n'+task['guidelines'])
                result=agent.chat(prompt)
                answer=result.response
                record=dict(task_id=task['task_id'],model=model,public_answer=task['answer'],
                    prediction=answer,exact_casefold_agreement=answer.strip().casefold()==task['answer'].strip().casefold(),
                    actions=[asdict(action)for action in result.actions],inner_steps=result.inner_steps,
                    scoring_scope='Exact casefold calibration diagnostic, not official DABStep score')
                handle.write(json.dumps(record,ensure_ascii=False)+'\n');handle.flush()
                print(task['task_id'],record['exact_casefold_agreement'],answer[:150],flush=True)
            finally:python.reset()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('model');p.add_argument('--run-id',required=True)
    p.add_argument('--endpoint',required=True)
    a=p.parse_args();main(a.root,a.model,a.run_id,a.endpoint)
