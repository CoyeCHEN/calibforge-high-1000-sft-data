import unittest
from unittest.mock import Mock,patch
from lagent.schema import ActionReturn,ActionStatusCode
from isolated_kernel import IsolatedIPython,IPythonInterpreter


class KernelTimeoutTest(unittest.TestCase):
    def interpreter(self):
        value=object.__new__(IsolatedIPython)
        value.initialize=Mock()
        value.kernel_manager=Mock()
        value.kernel_manager.is_alive.return_value=True
        value.kernel_client=Mock()
        return value

    def test_failed_python_drains_before_next_cell(self):
        value=self.interpreter();result=ActionReturn(args={},state=ActionStatusCode.API_ERROR,errmsg='Timeout')
        with patch.object(IPythonInterpreter,'__call__',return_value=result):
            self.assertIs(value('sleep'),result)
        value.kernel_client.wait_for_ready.assert_called_once_with(timeout=10)

    def test_terminal_timeout_becomes_an_error_only_after_ready(self):
        value=self.interpreter();value.kernel_client.execute_interactive.side_effect=TimeoutError('no output')
        with self.assertRaisesRegex(ValueError,'interrupted and ready'):value.terminal('bash',{'command':'sleep 200'})
        value.kernel_manager.interrupt_kernel.assert_called_once()
        value.kernel_client.wait_for_ready.assert_called_once_with(timeout=10)

    def test_unresponsive_kernel_remains_infrastructure_failure(self):
        value=self.interpreter();value.kernel_client.execute_interactive.side_effect=TimeoutError('no output')
        value.kernel_client.wait_for_ready.side_effect=RuntimeError('kernel unavailable')
        with self.assertRaisesRegex(RuntimeError,'kernel unavailable'):value.terminal('bash',{'command':'sleep 200'})


if __name__=='__main__':unittest.main()
