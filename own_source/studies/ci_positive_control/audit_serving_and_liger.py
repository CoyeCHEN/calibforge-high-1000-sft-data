"""Offline actual serving preprocessing, or training Liger binding (separate envs)."""
import functools
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

A=Path('/ssd/chenkeyi/qwen35-ci8197-implementation-audit-20261109')

def identify(f):
    result=[]
    while True:
        if isinstance(f,functools.partial):result.append(dict(partial=f.keywords));f=f.func;continue
        code=getattr(f,'__code__',None)
        result.append(dict(name=getattr(f,'__qualname__',type(f).__name__),file=code.co_filename if code else None,line=code.co_firstlineno if code else None))
        if not hasattr(f,'__wrapped__'):break
        f=f.__wrapped__
    return result

def main():
    if sys.argv[1]=='serving':
        from transformers import AutoTokenizer
        from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest
        from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
        tok=AutoTokenizer.from_pretrained('/ssd/chenkeyi/qwen35-ci-positive-control-20261108/tb21-after-ci/serving-view-epoch2',local_files_only=True)
        h=OpenAIServingChat.__new__(OpenAIServingChat);h.use_dpsk_v32_encoding=False
        h.template_manager=SimpleNamespace(jinja_template_content_format='openai',chat_template_name=None);h.tokenizer_manager=SimpleNamespace(tokenizer=tok)
        h.is_gpt_oss=False;h.reasoning_parser='qwen3';h.tool_call_parser='qwen3_coder'
        rows=[]
        for n,r in enumerate(json.loads((A/'eval-histogram-http.json').read_text())):
            req=ChatCompletionRequest(**r['request']);tools=[t.model_dump()for t in req.tools]
            inner_only=h._apply_jinja_template(req,tools,is_multimodal=False)
            actual=h._process_messages(req,is_multimodal=False)
            # Full router sorts recursively and excludes unset optional schema defaults.
            reference_tools=json.loads(json.dumps(r['request']['tools'],sort_keys=True))
            messages=json.loads(json.dumps(r['request']['messages']))
            for m in messages:
                for c in m.get('tool_calls')or []:
                    if isinstance(c['function']['arguments'],str):c['function']['arguments']=json.loads(c['function']['arguments'])
            expected=tok.apply_chat_template(messages,tools=reference_tools,enable_thinking=False,add_generation_prompt=True,tokenize=True,return_dict=False)
            ids=actual.prompt_ids;diff=next((i for i,(a,b)in enumerate(zip(ids,expected))if a!=b),None)
            row=dict(at=r['at'],response_id=r['response']['id'],expected_ids=expected,actual_preprocessor_ids=ids,
              ids_equal=ids==expected,first_difference=diff,actual_response_prompt_tokens=r['response'].get('usage',{}).get('prompt_tokens'),
              preprocessor_tokens=len(ids),inner_template_only_tokens=len(inner_only.prompt_ids),
              inner_vs_router_first_difference=next((i for i,(a,b)in enumerate(zip(inner_only.prompt_ids,ids))if a!=b),None),
              full_router_tools=reference_tools,inner_only_tools=tools,tool_observations=sum(m['role']=='tool'for m in messages),
              traceback_occurrences=sum('AttributeError' in str(m.get('content'))for m in messages if m['role']=='tool'))
            rows.append(row);(A/f'eval-attempt{n+1}-prefix.txt').write_text(tok.decode(ids,skip_special_tokens=False))
        src=Path(inspect.getfile(OpenAIServingChat))
        result=dict(kind='EVALUATION_CASE_NOT_TRAINING_ROW',mode='offline actual installed SGLang preprocessing; no generated model requests; historical worker input IDs unavailable',
          python=sys.executable,source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),rows=rows)
        (A/'SERVING_PREFIX_RESULT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
        print([(r['ids_equal'],r['preprocessor_tokens'],r['actual_response_prompt_tokens'])for r in rows],flush=True)
    elif sys.argv[1]=='liger':
        from transformers import AutoConfig
        from transformers.models.qwen3_5 import modeling_qwen3_5 as q
        from llamafactory.model.model_utils.liger_kernel import apply_liger_kernel
        from llamafactory.hparams import ModelArguments
        names=['Qwen3_5ForConditionalGeneration','Qwen3_5ForCausalLM','Qwen3_5RMSNorm','Qwen3_5MLP']
        before={n:identify(getattr(q,n).forward)for n in names}
        apply_liger_kernel(AutoConfig.from_pretrained('/data/models/Qwen/Qwen3.5-4B',local_files_only=True),
          ModelArguments(model_name_or_path='/data/models/Qwen/Qwen3.5-4B',enable_liger_kernel=True),True,False)
        after={n:identify(getattr(q,n).forward)for n in names}
        instance=object.__new__(q.Qwen3_5ForConditionalGeneration)
        f=functools.partial(q.Qwen3_5ForConditionalGeneration.forward.__get__(instance),skip_logits=True)
        result=dict(kind='RECONSTRUCTED_BINDING_NO_CUDA_EXECUTION',before=before,official_LF_liger_after=after,
          production_callback_forward=identify(f))
        (A/'LIGER_BINDINGS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
