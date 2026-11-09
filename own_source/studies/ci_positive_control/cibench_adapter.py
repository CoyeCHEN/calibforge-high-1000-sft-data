"""Native function transport for the official CIBench CIReAct/CodeAgent.

The official agent loop, interpreter and evaluator remain upstream code. The
adapter translates messages/tool calls, without answer or correction injection.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

from lagent.actions.base_action import BaseAction
from lagent.schema import ActionReturn, ActionStatusCode
from lagent.agents.react import ReActProtocol
from opencompass.models.base_api import BaseAPIModel
from opencompass.models.lagent import CodeAgent
from opencompass.lagent.agents.react import CIReAct
from isolated_kernel import IsolatedIPython
from tool_contract import terminal_tools


class NativeAssistant(str):
    """An actual transport packet, distinct from agent-generated plain text."""


class NativeTransport(BaseAPIModel):
    def __init__(self, root, endpoint, model, **kwargs):
        super().__init__(path=model, max_seq_len=262144, retry=0, **kwargs)
        self.root, self.endpoint = Path(root), endpoint
        self.tools = json.loads((self.root / "configs/tools.json").read_text())
        self.log = Path(os.environ["CI_MODEL_REQUEST_LOG"])

    def request(self, messages, max_out_len):
        payload = dict(model=self.path, messages=messages, tools=self.tools,
                       temperature=0., top_p=1., top_k=-1, min_p=0.,
                       presence_penalty=0., frequency_penalty=0., repetition_penalty=1.,
                       max_tokens=max_out_len, stream=False, seed=42,
                       chat_template_kwargs=dict(enable_thinking=False))
        request = urllib.request.Request(self.endpoint + "/v1/chat/completions",
                                         data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        started = time.time()
        with urllib.request.urlopen(request, timeout=180) as response:
            result = json.load(response)
        record = dict(at=started, seconds=time.time()-started, request=payload, response=result)
        with self.log.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        message=result["choices"][0]["message"]
        value=NativeAssistant(json.dumps(message,ensure_ascii=False))
        value.message=message
        return value

    def generate_from_template(self, prompt, max_out_len=512, **kwargs):
        return self.request(prompt, max_out_len)

    def generate(self, inputs, max_out_len=512, **kwargs):
        return [self.request(prompt, max_out_len) for prompt in inputs]


class NativeProtocol(ReActProtocol):
    def __init__(self):
        super().__init__()
        self.last_call = None

    def format(self, chat_history, inner_step, action_executor, force_stop=False):
        # The four behavioral guidelines are the official CIBench guidelines;
        # syntax is supplied by native function schemas instead of ReAct text.
        result = [dict(role="system", content=(
            "You are an assistant who can utilize external tools.\n"
            "1. Always use code interpreter to solve the problem.\n"
            "2. Use the declared Python tool to execute valid Python code.\n"
            "3. The generated codes will be executed in an ipython manner and the results will be cached.\n"
            "4. Your responded code should always be simple and only solves the problem in current step."))]
        for message in chat_history + inner_step:
            if message["role"] == "user":
                result.append(deepcopy(message))
            elif message["role"] == "assistant":
                content=message["content"]
                if isinstance(content,NativeAssistant):
                    value = content.message
                    result.append({key:value[key] for key in ("role","content","tool_calls") if value.get(key) is not None})
                else:
                    # CIReAct itself appends a plain assistant response after
                    # exhausting its retries. It is text, not a model packet.
                    result.append(dict(role="assistant",content=content))
            elif message["role"] == "system":
                result.append(json.loads(message["content"]))
            else:
                raise ValueError("Unexpected CIBench history role")
        if force_stop:
            raise ValueError("No unregistered force-stop prompt")
        return result

    def parse(self, response, action_executor):
        if not isinstance(response,NativeAssistant):
            raise TypeError("Model transport must return an explicit native packet")
        message = response.message
        calls = message.get("tool_calls") or []
        if len(calls) != 1:
            self.last_call = None
            return message.get("content") or "", "NoAction", response
        call = calls[0]
        self.last_call = call
        function = call["function"]
        args = function["arguments"]
        if isinstance(args, str):
            args = json.loads(args)
        if function["name"] == "add_and_execute_jupyter_code_cell":
            return message.get("content") or "", "IPythonInterpreter", dict(command=args["code"], timeout=120)
        if not action_executor.is_valid(function['name']):
            # Native names are never guessed or aliased. Upstream's invalid
            # action accepts an error string, not the unknown tool's kwargs.
            return message.get('content') or '', action_executor.invalid_action.name, (
                'Unknown tool name: '+function['name'])
        return message.get("content") or "", function["name"], args

    def format_response(self, action_return):
        if self.last_call is None:
            # No fabricated tool return: this is a parser error from the
            # official NoAction, represented as an explicit user-visible error.
            return json.dumps(dict(role="user", content=action_return.errmsg or "No valid tool call was generated."))
        content = action_return.errmsg or (action_return.result or {}).get("text", "")
        return json.dumps(dict(role="tool", tool_call_id=self.last_call["id"], content=content), ensure_ascii=False)


class HarnessAction(BaseAction):
    def __init__(self, tool, interpreter):
        super().__init__(name=tool["name"], description=tool["description"])
        self.tool, self.interpreter = tool, interpreter

    def __call__(self, **args):
        from jsonschema import validate,ValidationError
        try:
            validate(args, self.tool["parameters"])
            result = self.interpreter.terminal(self.name,args)
            text = "\n".join(v["text"] for v in result["content"] if v["type"] == "text")
            if result.get("isError"):
                return ActionReturn(args=args, type=self.name, result=dict(text=text),
                                    state=ActionStatusCode.API_ERROR, errmsg=text)
            return ActionReturn(args=args, type=self.name, result=dict(text=text))
        except (ValueError,ValidationError) as error:
            return ActionReturn(args=args, type=self.name, state=ActionStatusCode.API_ERROR,
                                errmsg=type(error).__name__ + ": " + str(error))


class FinalAnswerAction(BaseAction):
    def __init__(self):
        super().__init__(name="final_answer", description="Provide the final answer.")

    def __call__(self, answer):
        return ActionReturn(args=dict(answer=answer), type=self.name, result=dict(text=answer))


class NativeCodeAgent(CodeAgent):
    def __init__(self, root, endpoint, model, user_data_dir, **kwargs):
        self.root = root
        os.environ['CI_ROOT']=str(root)
        self.python = IsolatedIPython(user_data_dir=user_data_dir,name='IPythonInterpreter',trim_output=1024)
        tools, _ = terminal_tools(root, Path(os.environ["CI_EXECUTION_WORKDIR"]))
        actions = [self.python, FinalAnswerAction()] + [HarnessAction(t, self.python) for t in tools]
        # Upstream CodeAgent and CIReAct own the full execution loop and its
        # fixed three-call retry budget. This adds only concrete actions.
        super().__init__(llm=dict(type=NativeTransport, root=root, endpoint=endpoint, model=model),
                         agent_type=CIReAct, actions=[], protocol=dict(type=NativeProtocol), max_turn=3, **kwargs)
        for action in actions:
            self.agent._action_executor.add_action(action)

    def reset(self):
        self.agent._session_history = []
        self.python.reset()
        # Native kernel shutdown removes only this notebook's Docker container,
        # including its Python state, files and session-owned background jobs.
