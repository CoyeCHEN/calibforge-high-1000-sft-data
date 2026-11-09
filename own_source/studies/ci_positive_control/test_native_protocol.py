import copy
import json
import unittest
from unittest.mock import Mock
from cibench_adapter import HarnessAction,NativeAssistant,NativeProtocol


class NativeProtocolTest(unittest.TestCase):
    def packet(self):
        d=dict(role='assistant',content='Explain.',tool_calls=[dict(id='c1',type='function',
            function=dict(name='add_and_execute_jupyter_code_cell',arguments='{"code":"x = 41"}'))])
        packet=NativeAssistant(json.dumps(d));packet.message=d
        return packet

    def test_transport_packet_preserved(self):
        packet=self.packet();protocol=NativeProtocol()
        rendered=protocol.format([dict(role='user',content='Question'),dict(role='assistant',content=packet)],[],None)
        self.assertEqual(rendered[-1],packet.message)
        _,name,args=protocol.parse(packet,None)
        self.assertEqual((name,args),('IPythonInterpreter',dict(command='x = 41',timeout=120)))

    def test_actual_retry_exhaustion_response_is_plain_text(self):
        protocol=NativeProtocol();history=[dict(role='user',content='Question'),
            dict(role='assistant',content='对不起，我无法回答你的问题'),dict(role='user',content='Next question')]
        result=protocol.format(history,[],None)
        self.assertEqual(result[-2],history[-2])

    def test_deepcopy_retains_typed_packet(self):
        packet=self.packet();other=copy.deepcopy(packet)
        self.assertIsInstance(other,NativeAssistant)
        self.assertEqual(other.message,packet.message)

    def test_unknown_tool_is_an_error_observation_not_a_process_crash(self):
        from lagent.actions import ActionExecutor
        packet=self.packet();packet.message['tool_calls'][0]['function']['name']='add_and_execute_jupyter_code'
        executor=ActionExecutor([]);protocol=NativeProtocol()
        _,name,args=protocol.parse(packet,executor)
        result=executor(name,args)
        self.assertNotEqual(int(result.state),0)
        self.assertIn('Unknown tool name',result.errmsg)
        observation=json.loads(protocol.format_response(result))
        self.assertEqual(observation['tool_call_id'],'c1')
        self.assertEqual(observation['role'],'tool')


class NativeToolStatusTest(unittest.TestCase):
    def action(self, result):
        interpreter = Mock()
        interpreter.terminal.return_value = result
        tool = dict(name='bash', description='Execute a Bash command.',
                    parameters=dict(type='object', properties=dict(command=dict(type='string')),
                                    required=['command']))
        return HarnessAction(tool, interpreter)

    def test_native_bash_error_does_not_end_official_retry_loop_as_success(self):
        # Shape and text from the real exit-7 native tool canary. The upstream
        # CIReAct loop decides whether to retry from ActionReturn.state.
        text = 'expected-error\nCommand exited with code 7.'
        action = self.action(dict(content=[dict(type='text', text=text)],
                                  details=dict(exitCode=7), isError=True))
        result = action(command='printf expected-error; exit 7')
        self.assertNotEqual(int(result.state), 0)
        self.assertEqual(result.errmsg, text)
        protocol = NativeProtocol()
        protocol.last_call = dict(id='error-call')
        observation = json.loads(protocol.format_response(result))
        self.assertEqual(observation, dict(role='tool', tool_call_id='error-call', content=text))

    def test_native_success_keeps_success_status_and_output(self):
        action = self.action(dict(content=[dict(type='text', text='42')],
                                  details=dict(exitCode=0), isError=False))
        result = action(command='printf 42')
        self.assertEqual(int(result.state), 0)
        self.assertEqual(result.result, dict(text='42'))


if __name__=='__main__':unittest.main()
