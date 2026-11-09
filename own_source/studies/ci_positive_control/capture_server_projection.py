"""Run with the frozen serving environment: capture its real preprocessing."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

from transformers import AutoTokenizer
from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items() if v is not None}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def main(root):
    root=Path(root)
    tokenizer=AutoTokenizer.from_pretrained('/data/models/Qwen/Qwen3.5-4B',local_files_only=True)
    tools=json.loads((root/'configs/tools.json').read_text())
    handler=OpenAIServingChat.__new__(OpenAIServingChat)
    handler.use_dpsk_v32_encoding=False
    handler.is_gpt_oss=False
    handler.reasoning_parser='qwen3'
    handler.tool_call_parser='qwen3_coder'
    handler.template_manager=SimpleNamespace(jinja_template_content_format='openai',chat_template_name=None)
    handler.tokenizer_manager=SimpleNamespace(tokenizer=tokenizer)
    rows=[];projected=None
    for line in (root/'data/replay-traces.jsonl').open():
        source=json.loads(line)['source'];messages=clean(source['messages'])
        for index in [1,3]:
            request=ChatCompletionRequest(model='qwen35-ci-m0',messages=messages[:index],tools=tools,
                max_tokens=1,chat_template_kwargs={'enable_thinking':False})
            # Observe the exact tools reaching the tokenizer after the real
            # router's exclude_unset and ordering, without duplicating them.
            with patch.object(tokenizer,'apply_chat_template',wraps=tokenizer.apply_chat_template) as render:
                result=handler._process_messages(request,is_multimodal=False)
                assert render.call_count==1, 'Expected one successful Qwen template application'
                served_tools=render.call_args.kwargs['tools']
            if projected is None:projected=served_tools
            assert json.dumps(served_tools,ensure_ascii=False)==json.dumps(projected,ensure_ascii=False)
            reference=tokenizer.apply_chat_template(messages[:index],tools=served_tools,
                enable_thinking=False,add_generation_prompt=True,tokenize=True,return_dict=False)
            assert result.prompt_ids==reference
            rows.append(dict(trace_id=source['id'],target_index=index,
                             template_applications=render.call_count,
                             actual_preprocessor_input_ids=result.prompt_ids))
    projection_path=root/'configs/sglang-template-tools.json'
    projection_path.write_text(json.dumps(projected,indent=2)+'\n')
    import inspect
    src=Path(inspect.getfile(OpenAIServingChat))
    result=dict(status='PASSED',serving_source=str(src),serving_source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        processing_entry='_process_messages',
        tool_projection_file_sha256=hashlib.sha256(projection_path.read_bytes()).hexdigest(),
        tool_projection='Exact tools passed by the full router to the tokenizer, including exclude_unset and ordering',
        logical_tools_unchanged=True,checks=rows)
    (root/'checks/actual-server-preprocessor.json').write_text(json.dumps(result,indent=2)+'\n')
    print('REAL_PREPROCESSOR_PASSED',len(rows),flush=True)


if __name__=='__main__':main(sys.argv[1])
