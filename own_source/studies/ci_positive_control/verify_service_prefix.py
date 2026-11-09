"""Compare native training prefixes with the real server tokenizer and route."""
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

from sft.qwen35 import load_tokenizer,digest
from tool_contract import messages,schemas


def request(route,payload):
    req=urllib.request.Request('http://127.0.0.1:18641'+route,data=json.dumps(payload).encode(),
                               headers={'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=120) as h:
        return json.load(h)


def main(root):
    root=Path(root);tok=load_tokenizer();reports=[]
    for line in (root/'data/replay-traces.jsonl').open():
        item=json.loads(line);source=item['source'];ms=messages(source['messages'],digest(source['id'])[:16]);tools=schemas(root,source['tools'])
        rendered_tools=json.loads((root/'configs/sglang-template-tools.json').read_text())
        # Both initial and post-observation prefixes for every real trajectory.
        for target_index in [1,3]:
            prefix=tok.apply_chat_template(ms[:target_index],tools=rendered_tools,tokenize=False,
                                           add_generation_prompt=True,enable_thinking=False)
            ids=tok.apply_chat_template(ms[:target_index],tools=rendered_tools,tokenize=True,
                                        return_dict=False,add_generation_prompt=True,enable_thinking=False)
            server=request('/tokenize',dict(model='qwen35-ci-m0',prompt=prefix,add_special_tokens=False))
            assert server['tokens']==ids
            generated=request('/v1/chat/completions',dict(model='qwen35-ci-m0',messages=ms[:target_index],tools=tools,
                temperature=0,top_p=1,max_tokens=1,stream=False,chat_template_kwargs=dict(enable_thinking=False)))
            assert generated['usage']['prompt_tokens']==len(ids),(len(ids),generated['usage'])
            reports.append(dict(trace_id=source['id'],target_index=target_index,prefix_tokens=len(ids),
                prefix_ids_sha256=digest(ids),server_tokenizer_ids_exact=True,
                actual_chat_prompt_tokens=generated['usage']['prompt_tokens'],
                request_messages_sha256=digest(ms[:target_index]),actual_response=generated))
    report=dict(status='PASSED',checks=reports,enable_thinking=False,
                template_sha256=digest(tok.chat_template),
                boundary='The real /tokenize IDs exactly match; actual chat route counts also match. Internal worker input-ID dump is not asserted.')
    (root/'checks/service-prefix.json').write_text(json.dumps(report,indent=2)+'\n')
    print('SERVICE_PREFIX_PASSED',len(reports),flush=True)


if __name__=='__main__':main(sys.argv[1])
