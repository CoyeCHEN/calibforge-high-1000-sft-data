"""Official IPython interpreter, provisioned with a standard Docker kernel."""
import ast
import json
import os
from pathlib import Path
import shlex
import subprocess
import uuid

from jupyter_client import KernelManager
from opencompass.lagent.actions.ipython_interpreter import IPythonInterpreter


class IsolatedIPython(IPythonInterpreter):
    _KERNEL_CLIENTS={}

    @staticmethod
    def start_kernel():
        root=Path(os.environ['CI_ROOT'])
        runtime=root/'runtime/kernels';runtime.mkdir(parents=True,exist_ok=True)
        connection=runtime/('kernel-'+uuid.uuid4().hex+'.json')
        manager=KernelManager(kernel_name='ci-docker-python',connection_file=str(connection))
        manager.start_kernel()
        client=manager.client()
        client.start_channels()
        client.wait_for_ready(timeout=60)
        return manager,client

    def terminal(self,name,args):
        self.initialize()
        # Jupyter's execute_reply user_expressions transports a structured
        # result without a separate RPC protocol or a second agent loop.
        code=('import kernel_tools as _ci_kernel_tools\n'
              '_ci_tool_result = await _ci_kernel_tools.execute('+repr(name)+','+repr(args)+')')
        try:
            reply=self.kernel_client.execute_interactive(code,timeout=150,
                user_expressions={'result':'__import__("json").dumps(_ci_tool_result)'},output_hook=lambda msg:None)
        except TimeoutError as error:
            self.kernel_manager.interrupt_kernel()
            self.kernel_client.wait_for_ready(timeout=10)
            raise ValueError('Terminal execution exceeded 150 seconds; kernel interrupted and ready') from error
        content=reply['content']
        if content['status']!='ok':
            raise ValueError(content.get('ename','KernelError')+': '+content.get('evalue',''))
        value=content['user_expressions']['result']
        if value['status']!='ok':raise RuntimeError('Kernel did not return the native tool result')
        return json.loads(ast.literal_eval(value['data']['text/plain']))

    def __call__(self,command,timeout=None):
        result=super().__call__(command,timeout)
        if not self.kernel_manager.is_alive():
            raise RuntimeError('The isolated Python environment exited; no model score is inferred')
        if int(result.state)!=0:
            # The native interpreter sends SIGINT on an IOPub timeout, then
            # returns before the kernel has drained its aborted request queue.
            # A native kernel-info handshake keeps the next cell from being
            # aborted or mistaken for output of the interrupted cell.
            self.kernel_client.wait_for_ready(timeout=10)
        return result

    def reset(self):
        if not self._initialized:return
        manager=self.kernel_manager
        name='qwen35-ci-kernel-'+Path(manager.connection_file).stem
        try:
            manager.shutdown_kernel(now=False)
        finally:
            self.kernel_client.stop_channels()
            self._KERNEL_CLIENTS.pop(os.getpid(),None)
            self._initialized=False
            # Inspect exact ownership before cleaning an unresponsive kernel.
            query=['docker','inspect',name,'--format','{{json .Config.Labels}}']
            inspected=subprocess.run(['sg','docker','-c',shlex.join(query)],capture_output=True,text=True,timeout=20)
            if inspected.returncode==0:
                labels=json.loads(inspected.stdout)
                if labels.get('chenkeyi.run')!='qwen35-ci-positive-control-20261108':
                    raise RuntimeError('Kernel container ownership differs')
                subprocess.run(['sg','docker','-c',shlex.join(['docker','rm','-f',name])],check=True,
                               capture_output=True,text=True,timeout=25)
            elif 'no such' not in inspected.stderr.casefold():
                raise RuntimeError('Cannot verify Docker kernel cleanup: '+inspected.stderr[-300:])
