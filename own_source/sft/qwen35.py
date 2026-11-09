"""Pinned native Qwen serialization and explicit labels, consumed through LF tokenized_path."""
import bisect
import copy
import hashlib
import json
import re
from pathlib import Path

import os
MODEL = os.environ.get('QWEN35_MODEL_PATH', 'Qwen/Qwen3.5-4B')
REVISION = '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
TEMPLATE_SHA = 'a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715'
MARKER = re.compile(r'<\|im_start\|>(system|user|assistant)\n')
RESERVED = ('<|im_start|>', '<|im_end|>')


def digest(value):
    if not isinstance(value, str):
        value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(value.encode()).hexdigest()


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=Path(MODEL).is_dir())
    if digest(tok.chat_template) != TEMPLATE_SHA:
        raise ValueError('Pinned official template changed')
    return tok


def text_content(content):
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if not isinstance(content, list) or any(p.get('type') != 'text' for p in content):
        raise ValueError('Non-text content requires separate multimodal validation')
    return ''.join(p['text'] for p in content)


def normalize_message(source):
    """Normalize documented Recorder/OpenAI fields, without editing text or special markers."""
    m = copy.deepcopy(source)
    if m['role'] == 'toolResult':
        m = dict(role='tool', tool_call_id=m['toolCallId'], content=text_content(m.get('content')))
    if m['role'] == 'assistant' and isinstance(m.get('content'), list):
        parts = m['content']
        if any(p['type'] not in ('text', 'thinking', 'toolCall') for p in parts):
            raise ValueError('Unsupported assistant block')
        m = dict(role='assistant', content=''.join(p['text'] for p in parts if p['type'] == 'text'))
        thinking = ''.join(p['thinking'] for p in parts if p['type'] == 'thinking')
        if thinking:
            m['reasoning_content'] = thinking
        calls = [dict(id=p['id'], type='function', function=dict(name=p['name'], arguments=p['arguments']))
                 for p in parts if p['type'] == 'toolCall']
        if calls:
            m['tool_calls'] = calls
    else:
        m['content'] = text_content(m.get('content'))
    for call in m.get('tool_calls', []):
        f = call['function']
        if isinstance(f['arguments'], str):
            f['arguments'] = json.loads(f['arguments'])
        if not isinstance(f['arguments'], dict):
            raise ValueError('Tool arguments must be a JSON object')
    if m.get('reasoning_content') and ('<think>' in m['content'] or '</think>' in m['content']):
        raise ValueError('Ambiguous separate and embedded thinking')
    if any(t in json.dumps(m, ensure_ascii=False) for t in RESERVED):
        raise ValueError('Message contains reserved role boundary tokens')
    return m


def validate_pairs(messages):
    pending = {}; seen = set(); linkage = []
    for index, m in enumerate(messages):
        if m['role'] not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('Unsupported message role')
        if m['role'] == 'tool':
            ident = m.get('tool_call_id')
            if ident not in pending:
                raise ValueError('Orphan or duplicate tool result')
            # Fixed Qwen tool XML omits IDs. Preserve source order and reject a
            # parallel response order that would make that projection ambiguous.
            if ident != next(iter(pending)):
                raise ValueError('Tool result order cannot be represented by fixed Qwen template')
            call_index, name = pending.pop(ident)
            linkage.append(dict(id=ident, name=name, assistant_index=call_index, result_index=index))
            continue
        if pending:
            raise ValueError('Assistant batch has missing tool results')
        if m['role'] == 'system' and any(x['role'] != 'system' for x in messages[:index]):
            raise ValueError('System message after conversation began')
        for c in m.get('tool_calls', []):
            ident = c.get('id')
            if not ident or ident in seen:
                raise ValueError('Missing or duplicate tool call ID')
            seen.add(ident); pending[ident] = (index, c['function']['name'])
    # The current target may request tools whose results are deliberately not in its input.
    if pending and messages[-1]['role'] != 'assistant':
        raise ValueError('Unresolved history tool calls')
    return linkage, sorted(pending)


def encode(tok, source_messages, tools=None, *, scope='current', cutoff=131072):
    if scope not in ('current', 'all_assistant'):
        raise ValueError('Unknown supervision scope')
    messages = [normalize_message(m) for m in source_messages]
    if not messages or messages[-1]['role'] != 'assistant':
        raise ValueError('A supervised example must end at an actual assistant target')
    linkage, pending = validate_pairs(messages)
    # The official template owns thinking, tool XML, schema order, EOS and message boundaries.
    rendered = tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=False,
                                       enable_thinking=True)
    native = tok.apply_chat_template(messages, tools=tools, tokenize=True, return_dict=False,
                                     add_generation_prompt=False, enable_thinking=True)
    enc = tok(rendered, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc['input_ids']
    if ids != native:
        raise AssertionError('Rendered token IDs differ from native official template')
    canonical_ids = ids
    generation_prefix = None
    generation_prefix_ids = None
    if scope == 'current':
        # Inference tokenizes the prompt before any completion exists. Encoding
        # the concatenated text can merge the final prompt newline with an empty
        # thinking target's newline (198 + 198 -> 271), changing the context on
        # which the first generated tokens are supervised. Keep the official
        # text, but tokenize at the real generation boundary.
        generation_prefix = tok.apply_chat_template(messages[:-1], tools=tools,
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        generation_prefix_ids = tok.apply_chat_template(messages[:-1], tools=tools,
            tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=True)
        if not rendered.startswith(generation_prefix):
            raise ValueError('OFFICIAL_GENERATION_PREFIX_TEXT_REQUIRES_REVIEW')
        first = tok(generation_prefix, add_special_tokens=False, return_offsets_mapping=True)
        rest = tok(rendered[len(generation_prefix):], add_special_tokens=False, return_offsets_mapping=True)
        if first['input_ids'] != generation_prefix_ids:
            raise AssertionError('Native generation prefix IDs differ')
        enc = dict(input_ids=first['input_ids'] + rest['input_ids'],
            offset_mapping=first['offset_mapping'] +
                [(a+len(generation_prefix), b+len(generation_prefix)) for a,b in rest['offset_mapping']])
        ids = enc['input_ids']
    if len(ids) > cutoff:
        raise ValueError(f'EXCEEDS_CUTOFF_NO_TRUNCATION:{len(ids)}>{cutoff}')
    spans = list(MARKER.finditer(rendered))
    assistants = [m for m in messages if m['role'] == 'assistant']
    assistant_spans = [(m, spans[i+1].start() if i+1 < len(spans) else len(rendered))
                       for i, m in enumerate(spans) if m.group(1) == 'assistant']
    if len(assistants) != len(assistant_spans):
        raise ValueError('Unexpected role boundaries in template output')
    labels = [-100] * len(ids); supervised_ranges = []
    starts = [a for a, b in enc['offset_mapping']]
    for ordinal, (source, (mark, end)) in enumerate(zip(assistants, assistant_spans)):
        # Detect official historical-thinking removal rather than silently accepting lost content.
        standalone = tok.apply_chat_template([dict(role='user', content='codec-reference'), source],
                                             tokenize=False, add_generation_prompt=False, enable_thinking=True)
        reference = standalone.split('<|im_start|>assistant\n', 1)[1]
        if rendered[mark.end():end] != reference:
            raise ValueError('OFFICIAL_TEMPLATE_HISTORY_PROJECTION_REQUIRES_REVIEW')
        if scope == 'current' and ordinal != len(assistants)-1:
            continue
        left = bisect.bisect_left(starts, mark.end())
        right = bisect.bisect_left(starts, end)
        target = rendered[mark.end():end]
        if scope == 'all_assistant':
            opener = tok.encode('<think>\n', add_special_tokens=False)
            if target.startswith('<think>\n') and ids[left:left+len(opener)] != opener:
                raise ValueError('ALL_ASSISTANT_GENERATION_PREFIX_REQUIRES_DECISION_SEGMENTATION')
        expected = tok.encode(target, add_special_tokens=False)
        if generation_prefix is not None:
            expected = (tok.encode(rendered[mark.end():len(generation_prefix)], add_special_tokens=False)
                        + tok.encode(rendered[len(generation_prefix):end], add_special_tokens=False))
        if left == right or ids[left:right] != expected:
            raise AssertionError('Assistant label boundary does not align with token IDs')
        labels[left:right] = ids[left:right]
        supervised_ranges.append([left, right])
    if not supervised_ranges or not any(x != -100 for x in labels):
        raise ValueError('No supervised tokens')
    row = dict(input_ids=ids, attention_mask=[1]*len(ids), labels=labels,
               images=[], videos=[], audios=[])
    evidence = dict(text_tokens=len(ids), loss_tokens=sum(x != -100 for x in labels),
                    input_sha256=digest(ids), labels_sha256=digest(labels),
                    source_messages_sha256=digest(source_messages), normalized_messages_sha256=digest(messages),
                    rendered_sha256=digest(rendered), tools_sha256=digest(tools), supervision_scope=scope,
                    supervised_ranges=supervised_ranges, linkage=linkage, target_pending_tool_ids=pending,
                    id_policy='IDs checked and retained here; fixed Qwen XML represents calls/results by order, without IDs.',
                    native_assistant_block_policy='Text blocks retain their text order; fixed Qwen template projects body before tool calls. Source block order remains in immutable source.',
                    tokenization_policy=('native_generation_prefix_then_completion' if scope == 'current'
                                         else 'official_complete_conversation'),
                    canonical_input_sha256=digest(canonical_ids),
                    generation_prefix_retokenized=ids != canonical_ids,
                    generation_prefix_tokens=(len(generation_prefix_ids) if generation_prefix_ids is not None else None),
                    generation_prefix_sha256=(digest(generation_prefix_ids) if generation_prefix_ids is not None else None),
                    truncation=False)
    return row, evidence


def encode_call(tok, call, *, cutoff=131072):
    if call.get('context_fidelity') in ('context_only', 'unknown', 'incomplete'):
        raise ValueError('Actual teacher request unavailable')
    payload=call.get('provider_payload')or{}
    if not isinstance(payload.get('messages'),list):
        raise ValueError('Recorded provider payload messages required')
    # Old canonical JSON sorted object keys. Parsing its normalized argument dicts
    # changes Qwen parameter order; the original payload's JSON argument strings
    # retain the teacher request order. Compare semantics, then use that payload.
    native=[normalize_message(m)for m in payload['messages']]
    canonical=[normalize_message(m)for m in call['request_messages']]
    if digest(native)!=digest(canonical):
        raise ValueError('Canonical context differs semantically from actual provider request')
    row,meta=encode(tok,payload['messages']+[call['assistant_response']],payload.get('tools'),cutoff=cutoff,scope='current')
    meta['request_basis']='recorded_provider_payload_preserving_argument_order'
    meta['canonical_context_semantically_equal']=True
    return row,meta


def encode_multiturn_task(tok, calls, *, cutoff=131072):
    """Supervise a complete recorded conversation once, without fabricated history.

    Source/reward approval belongs to the dataset gate. This verifies that every
    decision's real Qwen-encoded input/target is an exact prefix of the one saved
    multi-turn sample. Compaction or rewritten observations cannot silently
    become a different context for an earlier target.
    """
    ids=[call.get('call_id')for call in calls]
    if not ids or any(not ident for ident in ids)or len(set(ids))!=len(ids):
        raise ValueError('Missing or duplicate decision identity')
    if any(call.get('context_fidelity')!='provider_payload'for call in calls):
        raise ValueError('Every decision requires its actual provider payload')
    last=calls[-1];payload=last['provider_payload']
    row,meta=encode(tok,payload['messages']+[last['assistant_response']],payload.get('tools'),
                    scope='all_assistant',cutoff=cutoff)
    if len(meta['supervised_ranges'])!=len(calls):
        raise ValueError('Assistant targets do not match the recorded decision count')
    mapping=[]
    for call,span in zip(calls,meta['supervised_ranges']):
        prefix,evidence=encode_call(tok,call,cutoff=cutoff)
        if (prefix['input_ids']!=row['input_ids'][:len(prefix['input_ids'])]
            or evidence['supervised_ranges']!=[span]):
            raise ValueError('MULTITURN_CONTEXT_CHANGED_OR_COMPACTED:'+call['call_id'])
        mapping.append(dict(call_id=call['call_id'],supervised_range=span,
            original_decision_input_sha256=evidence['input_sha256'],
            original_decision_labels_sha256=evidence['labels_sha256']))
    meta.update(sample_unit='complete_recorded_multiturn_task',assistant_targets_once=True,
                original_request_prefixes_verified=True,decision_mapping=mapping)
    return row,meta
